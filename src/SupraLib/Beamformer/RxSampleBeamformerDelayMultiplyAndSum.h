// ================================================================================================
//
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Adrian Riedl
//      Email Adrian.Riedl@in.tum.de
//
// ================================================================================================

#ifndef SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM_H
#define SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM_H


#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra {

class RxSampleBeamformerDelayMultiplyAndSum {
    public:
        /*!
         * This beamforming function tries to adapt the DMAS algorithm from the 2D case to the 3D case.
         * Therefore, the algorithm iterates over the receive window and for each actual element,
         * the summation and multiplication is done on the subwindow starting
         * from the next right column and the next row below the actual element.
         * (The actual element stops before the subwindow size becomes 0)
         *
         * Implementation of the 2D Delay Multiply and Sum algorithm according to the paper
         * 'Signed Real-Time Delay Multiply and Sum Beamforming for Multispectral Photoacoustic Imaging'
         * (source: https://www.researchgate.net/publication/328335303_Signed_Real-Time_Delay_Multiply_and_Sum_Beamforming_for_Multispectral_Photoacoustic_Imaging)
         *
         */
        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
        static __device__ ResultType sampleBeamform3D(
            ScanlineRxParameters3D::TransmitParameters txParams,
            const RFType *RF,
            vec2T<uint32_t> elementLayout,
            uint32_t numReceivedChannels,
            uint32_t numTimesteps,
            const LocationType *x_elemsDTsh,
            const LocationType *z_elemsDTsh,
            LocationType scanline_x,
            LocationType scanline_z,
            LocationType dirX,
            LocationType dirY,
            LocationType dirZ,
            LocationType aDT,
            LocationType depth,
            vec2f invMaxElementDistance,
            LocationType speedOfSound,
            LocationType dt,
            int32_t additionalOffset,
            const WindowFunctionGpu *windowFunction,
            const WindowFunction::ElementType *functionShared
        )
        {
            float sample = 0.0f;
            float weightAcum = 0.0f;
            LocationType initialDelay = txParams.initialDelay;
            uint32_t txScanlineIdx = txParams.txScanlineIdx;
            int32_t numAcc = 0;

            for (int32_t elemIdxX = txParams.firstActiveElementIndex.x;
                 elemIdxX < txParams.lastActiveElementIndex.x - 1; elemIdxX++) {
                for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y;
                     elemIdxY < txParams.lastActiveElementIndex.y - 1; elemIdxY++) {
                    uint32_t elemIdx = elemIdxX + elemIdxY * elementLayout.x;
                    uint32_t channelIdx = elemIdx % numReceivedChannels;
                    LocationType x_elem = x_elemsDTsh[elemIdx];
                    LocationType z_elem = z_elemsDTsh[elemIdx];
                    vec2f elementScanlineDistance = {x_elem - scanline_x, z_elem - scanline_z};
                    LocationType delayfxElem = initialDelay +
                                               computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x,
                                                                       scanline_z, depth) + additionalOffset;
                    uint32_t delayfx = static_cast<uint32_t>(floor(delayfxElem));
                    float weightxElem = computeWindow3DShared(*windowFunction, functionShared,
                                                              elementScanlineDistance * invMaxElementDistance);
                    float RFxElem = RF[delayfx + channelIdx * numTimesteps +
                                       txScanlineIdx * numReceivedChannels * numTimesteps];
                    float RFxElemModified = (RFxElem < 0) ? ((-1) * std::sqrt(std::abs(RFxElem))) : (std::sqrt(RFxElem));
                    for (uint32_t elemIdxXShift = elemIdxX + 1;
                         elemIdxXShift < txParams.lastActiveElementIndex.x; elemIdxXShift++) {
                        for (uint32_t elemIdxYShift = elemIdxY + 1;
                             elemIdxYShift < txParams.lastActiveElementIndex.y; elemIdxYShift++) {
                            uint32_t elemIdxShift = elemIdxXShift + elemIdxYShift * elementLayout.x;
                            uint32_t channelIdxShift = elemIdxShift % numReceivedChannels;
                            LocationType x_elemShift = x_elemsDTsh[elemIdxShift];
                            LocationType z_elemShift = z_elemsDTsh[elemIdxShift];
                            vec2f elementScanlineDistanceShift = {x_elemShift - scanline_x, z_elemShift - scanline_z};
                            LocationType delayfxElemShift = initialDelay +
                                                            computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elemShift,
                                                                                    z_elemShift, scanline_x, scanline_z,
                                                                                    depth) + additionalOffset;
                            int32_t delayElemShift = static_cast<int32_t>(::floor(delayfxElemShift));
                            float weightxElemShift = computeWindow3DShared(*windowFunction, functionShared,
                                                                           elementScanlineDistanceShift *
                                                                           invMaxElementDistance);
                            float RFxElemShift = RF[delayElemShift + channelIdxShift * numTimesteps +
                                                    txScanlineIdx * numReceivedChannels * numTimesteps];
                            float RFxElemShiftedModified = (RFxElemShift < 0) ? ((-1) * std::sqrt(std::abs(RFxElemShift)))
                                                                              : (std::sqrt(RFxElemShift));
                            sample += RFxElemShiftedModified * RFxElemModified * weightxElemShift * weightxElem;

                            numAcc++;
                            weightAcum += weightxElemShift * weightxElem;
                        }
                    }
                }
            }
            return sample / numAcc * weightAcum;
        }

        /*!
         * This beamforming function considers the weights of the different elements.
         * Summation is done with weights of the corresponding elements.
         * The result is normalized by the number of accumulations and multiplied with the sum of the accumulated weights.
         *
         * Implementation of the 2D Delay Multiply and Sum algorithm according to the paper
         * 'Signed Real-Time Delay Multiply and Sum Beamforming for Multispectral Photoacoustic Imaging'
         * (source: https://www.researchgate.net/publication/328335303_Signed_Real-Time_Delay_Multiply_and_Sum_Beamforming_for_Multispectral_Photoacoustic_Imaging)
         *
         * @tparam interpolateRFlines
         * @tparam RFType
         * @tparam ResultType
         * @tparam LocationType
         * @param txParams
         * @param RF                                   Pointer to the raw data which needs to be beamformed
         * @param numTransducerElements
         * @param numReceivedChannels
         * @param numTimesteps
         * @param x_elemsDT
         * @param scanline_x
         * @param dirX
         * @param dirY
         * @param dirZ
         * @param aDT
         * @param depth
         * @param invMaxElementDistance
         * @param speedOfSound                          The speed of sound in this tissue
         * @param dt
         * @param additionalOffset
         * @param windowFunction
         * @return                                      The value of the beamformed raw data.
         */
        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
        static __device__ ResultType sampleBeamform2D(
            ScanlineRxParameters3D::TransmitParameters txParams,
            const RFType *RF,
            uint32_t numTransducerElements,
            uint32_t numReceivedChannels,
            uint32_t numTimesteps,
            const LocationType *x_elemsDT,
            LocationType scanline_x,
            LocationType dirX,
            LocationType dirY,
            LocationType dirZ,
            LocationType aDT,
            LocationType depth,
            LocationType invMaxElementDistance,
            LocationType speedOfSound,
            LocationType dt,
            int32_t additionalOffset,
            const WindowFunctionGpu *windowFunction
        )
        {
            float sample = 0.0f;
            float weightAcum = 0.0f;
            LocationType initialDelay = txParams.initialDelay;
            uint32_t txScanlineIdx = txParams.txScanlineIdx;
            int32_t numAcc = 0;

            for (int32_t elemIdxX = txParams.firstActiveElementIndex.x;
                 elemIdxX < txParams.lastActiveElementIndex.x - 1; elemIdxX++) {
                int32_t channelIdx = elemIdxX % numReceivedChannels;
                LocationType x_elem = x_elemsDT[elemIdxX];
                LocationType delayfxElem = initialDelay +
                                           computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth) +
                                           additionalOffset;
                int32_t delayElem = static_cast<int32_t>(floor(delayfxElem));
                float weightxElem = windowFunction->get((x_elem - scanline_x) * invMaxElementDistance);
                float RFxElem = RF[delayElem + channelIdx * numTimesteps +
                                   txScanlineIdx * numReceivedChannels * numTimesteps];
                float RFxElemModified = (RFxElem < 0) ? ((-1) * std::sqrt(std::abs(RFxElem))) : (std::sqrt(RFxElem));
                for (int32_t elemIdxXShift = elemIdxX + 1;
                     elemIdxXShift < txParams.lastActiveElementIndex.x; elemIdxXShift++) {
                    int32_t channelIdxShift = elemIdxXShift % numReceivedChannels;
                    LocationType x_elemShift = x_elemsDT[elemIdxXShift];
                    LocationType delayfxElemShift = initialDelay +
                                                    computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elemShift, scanline_x,
                                                                          depth) + additionalOffset;
                    int32_t delayElemShift = static_cast<int32_t>(floor(delayfxElemShift));
                    float weightxElemShift = windowFunction->get((x_elemShift - scanline_x) * invMaxElementDistance);
                    float RFxElemShift = RF[delayElemShift + channelIdxShift * numTimesteps +
                                            txScanlineIdx * numReceivedChannels * numTimesteps];
                    float RFxElemShiftedModified = (RFxElemShift < 0) ? ((-1) * std::sqrt(std::abs(RFxElemShift)))
                                                                      : (std::sqrt(RFxElemShift));
                    sample += RFxElemModified * RFxElemShiftedModified * weightxElem * weightxElemShift;

                    numAcc++;
                    weightAcum += weightxElemShift * weightxElem;
                }
            }
            return sample / numAcc * weightAcum;
        }
    };
}


#endif //SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM_H
