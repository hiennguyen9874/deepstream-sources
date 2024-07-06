#ifndef CVCORE_HEADPOSE_H_
#define CVCORE_HEADPOSE_H_

#include <cv/core/Array.h>
#include <cv/core/CameraModel.h>
#include <cv/core/Core.h>
#include <cv/core/MathTypes.h>

#include <memory>
#include <vector>

namespace cvcore {
namespace headpose {

/**
 * Parameters for HeadPose
 */
struct HeadPoseParams {
    size_t numLandmarks;                 /**< number of landmarks. */
    std::vector<double> facePoints3d;    /**< 3d points coordinates for face model. */
    std::vector<int> landmarks2dIndices; /**< indices for relevant landmarks. */
};

/**
 * Interface for running HeadPose.
 */
class CVCORE_API HeadPose {
public:
    /**
     * Constructor of HeadPose.
     * @param cameraParams camera intrinsics params for Headpose.
     */
    HeadPose(const CameraIntrinsics &cameraParams);

    /**
     * Constructor of HeadPose.
     * @param params custom params for HeadPose.
     * @param cameraParams camera intrinsics params for Headpose.
     */
    HeadPose(const HeadPoseParams &params, const CameraIntrinsics &cameraParams);

    /**
     * Destructor of HeadPose.
     */
    ~HeadPose();

    /**
     * Main interface to get head pose results.
     * @param outputPose output head pose results.
     * @param inputLandmarks input landmarks 2d coordinates of the face.
     */
    void execute(Pose3d &outputPose, const Array<Vector2d> &inputLandmarks);

private:
    /**
     * Implementation of HeadPose.
     */
    struct HeadPoseImpl;
    std::unique_ptr<HeadPoseImpl> m_pImpl;
};

} // namespace headpose
} // namespace cvcore

#endif
