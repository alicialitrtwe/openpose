// synchronous mode for video directory

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(video_dir,                "video_path/",
    "Use a video file instead of the camera. Use `examples/media/video.avi` for our default example video.");
DEFINE_string(camera_parameter_path,    "models/cameraParameters/flir/",
    "String with the folder where the camera parameters are located. If there is only 1 XML file (for single"
    " video, webcam, or images from the same camera), you must specify the whole XML file path (ending in .xml).");

// This worker will just read and return all the basic image file formats in a directory
class WUserInput : public op::WorkerProducer<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
public:
    WUserInput(const std::string& directoryPath, const std::string& cameraParameterPath):
        mVideoFiles{op::getFilesOnDirectory(directoryPath, op::Extensions::Videos)},
        mFrameCounter{0ull}
    {
        if (mVideoFiles.empty())
            op::error("No video found on: " + directoryPath, __LINE__, __FUNCTION__, __FILE__);

        for(const auto& mVideoFile: mVideoFiles) {
            mVideoReaders.emplace_back(std::make_shared<op::VideoReader>(mVideoFile));
            std::cout << mVideoFile;
        }

        // Create CameraParameterReader
        mCameraParameterReader.readParameters(cameraParameterPath);
    }

    void initializationOnThread() {}

    std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> workProducer()
    {
        try
        {
            // std::lock_guard<std::mutex> g(lock);
            if (mQueuedElements.empty())
            {
                // Camera parameters
                const std::vector<op::Matrix> &cameraMatrices = mCameraParameterReader.getCameraMatrices();
                const std::vector<op::Matrix> &cameraIntrinsics = mCameraParameterReader.getCameraIntrinsics();
                const std::vector<op::Matrix> &cameraExtrinsics = mCameraParameterReader.getCameraExtrinsics();
                const auto matrixesSize = cameraMatrices.size();
                // More sanity checks
                if (cameraMatrices.size() < 2)
                    op::error("There is less than 2 camera parameter matrices.",
                              __LINE__, __FUNCTION__, __FILE__);
                if (cameraMatrices.size() != cameraIntrinsics.size() || cameraMatrices.size() != cameraExtrinsics.size())
                    op::error("Camera parameters must have the same size.", __LINE__, __FUNCTION__, __FILE__);

                for (auto datumIndex = 0; datumIndex < matrixesSize; ++datumIndex)
                {
                    // Create new datum
                    // op::opLog(datumIndex);
                    auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
                    datumsPtr->emplace_back();
                    auto& datumPtr = datumsPtr->back();
                    datumPtr = std::make_shared<op::Datum>();

                    //auto &datumPtr = datumsPtr->at(datumIndex);
                    //datumPtr = std::make_shared<op::Datum>();
                    // Fill datum
                    const auto frame = mVideoReaders[datumIndex]->getFrame();
                    datumPtr->cvInputData = frame;
                    datumPtr->frameNumber = mFrameCounter;
                    if (matrixesSize > 1) {
                        datumPtr->subId = datumIndex;
                        datumPtr->subIdMax = matrixesSize - 1;
                        datumPtr->cameraMatrix = cameraMatrices[datumIndex];
                        datumPtr->cameraExtrinsics = cameraExtrinsics[datumIndex];
                        datumPtr->cameraIntrinsics = cameraIntrinsics[datumIndex];
                    }
                    // If empty frame -> return nullptr
                    if (datumPtr->cvInputData.empty())
                    {
                        // Close program when empty frame
                        op::opLog("Empty frame detected, closing program.", op::Priority::High);
                        this->stop();
                        return nullptr;
                    }

                    mQueuedElements.push(datumsPtr);
                }
                ++mFrameCounter;
            }

            auto Datums = mQueuedElements.front();
            mQueuedElements.pop();
            // Return result
            return Datums;

        }

        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }



private:
    unsigned long long mFrameCounter;
    op::CameraParameterReader mCameraParameterReader;
    std::vector<std::string> mVideoFiles;
    std::vector<std::shared_ptr<op::VideoReader>> mVideoReaders;
    std::queue<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>> mQueuedElements;
    std::mutex lock;
};

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Initializing the user custom classes
        // Frames producer (e.g., video, webcam, ...)
        auto wUserInput = std::make_shared<WUserInput>(FLAGS_video_dir, FLAGS_camera_parameter_path);
        // Add custom processing
        const auto workerInputOnNewThread = true;
        opWrapper.setWorker(op::WorkerType::Input, wUserInput, workerInputOnNewThread);

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapper.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Required flags to enable 3-D
        FLAGS_3d = true;
        FLAGS_number_people_max = 1;
        FLAGS_3d_min_views = 3;
        FLAGS_output_resolution = "320x256"; // Optional, but otherwise it gets too big to render in real time
        // FLAGS_3d_views = X; // Not required because it only affects OpenPose producers (rather than custom ones)

        // OpenPose wrapper
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper;
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.exec();

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
