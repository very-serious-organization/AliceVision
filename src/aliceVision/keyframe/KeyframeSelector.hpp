// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <limits>

namespace aliceVision {
namespace image {

template<typename T>
class Image;

} // namespace image

namespace keyframe {


class KeyframeSelector
{
public:    

    /**
     * @brief Process media paths and build a list of selected frames using a smart method based on optical flow estimation.
     * @param[in] mediaPaths For each camera, give the media file name (path)
     */
    void processSmart(const std::vector<std::string>& mediaPaths);

    /**
     * @brief Process media paths and build a list of selected frames using a regular sampling over time.
     * @param[in] mediaPaths for each camera, give the media file name (path)
     */
    void processRegular(const std::vector<std::string>& mediaPaths);

    /**
     * @brief Write selected frames indices.
     * @param[in] outputFolder Folder for output images
     * @param[in] mediaPaths For each camera, give the media file name (path)
     * @param[in] brands For each camera, give the brand name
     * @param[in] models For each camera, give the models name
     * @param[in] mmFocals For each camera, give the focal in mm
     */
    bool writeSelection(const std::string& outputFolder, const std::vector<std::string>& mediaPaths,
                        const std::vector<std::string>& brands, const std::vector<std::string>& models,
                        const std::vector<float>& mmFocals);

    /**
     * @brief Compute the sharpness score for every frame of the video and export it in a CSV file.
     * @param[in] mediaPaths For each camera, give the media file name (path)
     * @param[in] folder The folder where you want to save the statistics file
     * @param[in] filename The filename of the sharpness score file
     * @return False it cannot open the file, true if it succeed
     */
    bool exportSharpnessToFile(const std::vector<std::string>& mediaPaths, const std::string& folder,
                               const std::string& filename = "sharpness.csv") const;

    /**
     * @brief Compute the optical flow score for every frame of the video and export it in a CSV file.
     * @param[in] mediaPaths For each camera, give the media file name (path)
     * @param[in] folder The folder where you want to save the statistics file
     * @param[in] filename The filename of the sharpness score file
     * @return False it cannot open the file, true if it succeed
     */
    bool exportFlowToFile(const std::vector<std::string>& mediaPaths, const std::string& folder,
                          const std::string& filename = "flow.csv") const;


    /**
     * @brief Compute the sharpness and flow scores for the input media paths.
     * @param[in] mediaPath For each camera, give the media file name (path)
     * @param[in] folder The output folder
     * @param[in] rescale True if the score computation is also to be performed on rescaled images
     * @param[in] flowOnBorders True if the optical flow score is computed on the frame's borders
     * @param[in] flowByCell True if the optical flow score is computed cell by cell in the frame
     */
    bool computeScores(const std::vector<std::string>& mediaPaths, const std::string& folder, bool rescale = false,
        bool flowOnBorders = false, bool flowByCell = false);

    /**
     * @brief Based on the computed scores, select frames that are deemed relevant.
     * @param[in] refine True if the initial frame selection is to be refined, false otherwise
     */
    void selectFrames(bool refine = true);

    /**
     * @brief Select frames from the sequences using the motion and sharpness information.
     * @param[in] pxDisplacement The minimum of displaced pixels between two keyframes, in percent
     * @param[in] minFrames The minimum number of selected frames
     * @param[in] maxFrames The maximum number of selected frames
     */
    void selectFrames(float pxDisplacement, unsigned int minFrames = 10, unsigned int maxFrames = 2000);

    /**
     * @brief Select frames based on their borders' motion scores.
     * @return A vector containing the ID of the selected frames
     */
    std::vector<unsigned int> selectFramesWithMotion();

    /**
     * @brief Refine the selection based of the borders' motion scores using sharpness scores.
     * @param[in] selectedFrames A vector containing the ID of the frames selected based on their motion scores
     * @return A vector containing the ID of the selected frames after refining
     */
    std::vector<unsigned int> refineFrameSelection(const std::vector<unsigned int>& selectedFrames);

    /**
     * @brief Export all the sharpness and flow scores (full resolution and rescaled) to a CSV file.
     * @param[in] outputFolder The folder in which the CSV file will be written
     * @param[in] flowOnBorders True if the optical flow scores have been computed on each frame's borders
     */
    bool exportAllScoresToFile(const std::string& outputFolder, bool flowOnBorders) const;

    bool exportFlowByCellScores(const std::string& outputFolder) const;

    /**
     * @brief Export score vectors to a CSV file.
     * @param[in] scores A vector containing at least one score vector
     * @param[in] folder Folder in which the CSV file will be written
     * @param[in] finelame Name of the CSV file that will be written
     * @param[in] header Header for the CSV file that will be written
     */
    bool exportScoresToFile(const std::vector<std::vector<double>>& scores, const std::string& folder,
                        const std::string& filename, const std::string& header) const;

    /**
     * @brief Set the mininum frame step for the processing algorithm.
     * @param[in] frameStep minimum number of frames between two keyframes
     */
    void setMinFrameStep(unsigned int frameStep)
    {
        _minFrameStep = frameStep;
    }

    /**
     * @brief Set the maximum number of output frames for the processing algorithm.
     * @param[in] nbFrame Maximum number of output frames (if 0, no limit)
     */
    void setMaxOutFrame(unsigned int nbFrame)
    {
        _maxOutFrame = nbFrame;
    }

    /**
     * @brief Get the minimum frame step for the processing algorithm.
     * @return Minimum number of frames between two keyframes
     */
    unsigned int getMinFrameStep() const
    {
        return _minFrameStep;
    }

    /**
     * @brief Get the maximum number of output frames for the processing algorithm.
     * @return maximum number of output frames (if 0, no limit)
     */
    unsigned int getMaxOutFrame() const
    {
        return _maxOutFrame;
    }

private:

    /// Minimum number of frame between two keyframes
    unsigned int _minFrameStep = 12;
    /// Maximum number of output frame (0 = no limit)
    unsigned int _maxOutFrame = 0;

    /// List of selected frames
    std::vector<unsigned int> _selected;

    /// Sharpness scores for each frame (full res)
    std::vector<double> _sharpnessScores;
    /// Sharpness scores for each frame (rescaled)
    std::vector<double> _sharpnessScoresRescaled;
    /// Optical flow scores for each frame (full res)
    std::vector<double> _flowScores;
    /// Optical flow scores for each frame (rescaled)
    std::vector<double> _flowScoresRescaled;
    /// Optical flow scores for each frame's top border
    std::vector<double> _flowScoresOnTopBorder;
    /// Optical flow scores for each frame's bottom border
    std::vector<double> _flowScoresOnBottomBorder;
    /// Optical flow scores for each frame's left border
    std::vector<double> _flowScoresOnLeftBorder;
    /// Optical flow scores for each frame's right border
    std::vector<double> _flowScoresOnRightBorder;

    /// Optical flow scores for each frame, computing flow in cells over the whole frame
    std::vector<double> _flowScoresByCell;

    /// Sharpness scores for each frame that has been selected based on its motion score
    const unsigned int _internalFrameClusterLimit = 24;
    const unsigned int _internalMinFrameStep = 24;
    const unsigned int _internalMaxFrames = 2000;

    unsigned int frameWidth = 0;
    unsigned int frameHeight = 0;
};

} // namespace keyframe 
} // namespace aliceVision
