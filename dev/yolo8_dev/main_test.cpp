
/*****************************************
* Includes
******************************************/
/*DRP-AI TVM[*1] Runtime*/
#include "MeraDrpRuntimeWrapper.h"
/*Pre-processing Runtime Header*/
#include "PreRuntime.h"
/*DRPAI Driver Header*/
#include <linux/drpai.h>
/*DFL process control*/
#include "dfl_proc.h"
/*Image control*/
#include "image_yolov8.h"
/*box drawing*/
#include "box.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <mutex>
#include <cstdint>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <hiredis.h>

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h11.h"
#include "cJSON.h"
}


#define CAMERA_WIDTH    1280
#define CAMERA_HEIGHT   720

#define FX  710.3011047053091
#define CX  636.197363211571
#define FY  708.8122718473162
#define CY  352.6950924984073

#define TAG_SIZE 0.04

static cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    FX, 0,  CX, // fx, 0, cx
    0,  FY, CY, // 0, fy, cy
    0,  0,  1   );
static cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
    -0.3503279178080724, 0.1207413155993408, -0.0004422109104798752, -7.901348306332831e-05, -0.01930867871482221); // k1, k2, p1, p2, k3

static cv::VideoCapture camera_cap(0, cv::CAP_V4L2);
static redisContext* redis_conn = nullptr;
static bool debug_img = false;


int setup_camera() {
    // Check camera opened
    if (!camera_cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    // Set format to YUYV (YUV422)
    if (!camera_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','U','Y','V'))) {
        std::cerr << "Error: Could not set YUYV format." << std::endl;
        return -1;
    }
    // Set resolution and FPS
    camera_cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    camera_cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    // cap.set(cv::CAP_PROP_FPS, 30);
    return 0;
}


int main(int argc, char** argv) {
    // ======= [Argument] =======
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug_img") {
            debug_img = true;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            return -1;
        }
    }
    // ======= [Setup] =======
    int setup_ret = 0;
    // camera
    setup_ret = setup_camera();
    if(setup_ret != 0) {
        std::cerr << "Error: Setup camera failed" << std::endl;
        return setup_ret;
    }
    // redis
    redis_conn = redisConnect("127.0.0.1", 6379);
    if (redis_conn == NULL || redis_conn->err) {
        if (redis_conn) {
            std::cerr << "Redis connection error: " << redis_conn->errstr << std::endl;
            redisFree(redis_conn);
        } else {
            std::cerr << "Redis connection error: can't allocate context" << std::endl;
        }
        return -1;
    }

    cv::Mat frame, undistorted_frame, gray;
    uint32_t frame_id = 0;
    redisReply *reply;

    apriltag_family_t *tag_family = tag36h11_create();;
    apriltag_detector_t *tag_detector = apriltag_detector_create();
    apriltag_detector_add_family(tag_detector, tag_family);
    // tag_detector->nthreads = 1;
    // tag_detector->refine_edges = false;
    // tag_detector->debug = false;
    // tag_detector->quad_sigma = 0.8;
    // tag_detector->quad_decimate = 1;

    while (true) {
        if (!camera_cap.read(frame)) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }
        // ======= [frame available] =======
        // get time
        int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        // print info
        std::cout << "Frame: " << frame_id++ 
                  << ", shape: " << frame.size() << ", channels: " << frame.channels() << "\n";
        // undistort
        cv::undistort(frame, undistorted_frame, cameraMatrix, distCoeffs);
        // ======= [apriltag] =======
        // preprocess and detect
        cvtColor(undistorted_frame, gray, cv::COLOR_BGR2GRAY);
        image_u8_t tag_img = {gray.cols, gray.rows, gray.cols, gray.data};
        zarray_t *detections = apriltag_detector_detect(tag_detector, &tag_img);
        // process apriltag result
        cJSON* json_arr = cJSON_CreateArray();
        for (int idx = 0; idx < zarray_size(detections); idx++) {
            apriltag_detection_t *det;
            zarray_get(detections, idx, &det);
            apriltag_detection_info_t tag_info = {
                .det = det,
                .tagsize = TAG_SIZE,
                .fx = FX,
                .fy = FY,
                .cx = CX,
                .cy = CY,
            };
            apriltag_pose_t pose;
            estimate_tag_pose(&tag_info, &pose);
            // create json
            cJSON* det_json = cJSON_CreateObject();
            cJSON_AddItemToArray(json_arr, det_json);
            // id
            cJSON_AddNumberToObject(det_json, "id", det->id);
            // center
            cJSON* c_json = cJSON_CreateArray();
            cJSON_AddItemToArray(c_json, cJSON_CreateNumber(det->c[0]));
            cJSON_AddItemToArray(c_json, cJSON_CreateNumber(det->c[1]));
            cJSON_AddItemToObject(det_json, "center", c_json);
            // corner
            cJSON* p_json = cJSON_CreateArray();
            for (int i = 0; i < 4; ++i) {
                cJSON* corner_json = cJSON_CreateArray();
                cJSON_AddItemToArray(corner_json, cJSON_CreateNumber(det->p[i][0]));
                cJSON_AddItemToArray(corner_json, cJSON_CreateNumber(det->p[i][1]));
                cJSON_AddItemToArray(p_json, corner_json);
            }
            cJSON_AddItemToObject(det_json, "corner", p_json);
            // pose R
            cJSON* R_json = cJSON_CreateArray();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cJSON_AddItemToArray(R_json, cJSON_CreateNumber(pose.R->data[i * 3 + j]));
                }
            }
            cJSON_AddItemToObject(det_json, "rotation", R_json);
            // pose t
            cJSON* t_json = cJSON_CreateArray();
            for (int i = 0; i < 3; ++i) {
                cJSON_AddItemToArray(t_json, cJSON_CreateNumber(pose.t->data[i]));
            }
            cJSON_AddItemToObject(det_json, "position", t_json);
            // time
            cJSON_AddNumberToObject(det_json, "timestamp", now_ns);
        }
        // redis send detection
        char* json_str = cJSON_PrintUnformatted(json_arr);
        reply = (redisReply*)redisCommand(redis_conn, "set detections %s", json_str);
        // release object
        freeReplyObject(reply);
        free(json_str);
        cJSON_Delete(json_arr);
        apriltag_detections_destroy(detections);
        // ======= [redis] =======
        // send debug img when flag set
        if(debug_img) {
            reply = (redisReply*)redisCommand(redis_conn, "set debug_img %b", undistorted_frame.data, CAMERA_HEIGHT * CAMERA_WIDTH * 3);
            freeReplyObject(reply);
        }
        // img id (must be last)
        reply = (redisReply*)redisCommand(redis_conn, "set img_id %u", frame_id);
        freeReplyObject(reply);
    }

    camera_cap.release();
    return 0;
}
