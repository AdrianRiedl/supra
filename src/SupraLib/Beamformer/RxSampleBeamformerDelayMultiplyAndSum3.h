// ================================================================================================
//
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Adrian Riedl
//      Email Adrian.Riedl@in.tum.de
//
// ================================================================================================

#ifndef SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM3_H
#define SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM3_H


#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra {

    class RxSampleBeamformerDelayMultiplyAndSum3 {
    public:
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
            return 0;
        }

        /*!
         * This beamforming function considers the weights of the different elements.
         * Summation is done without weights of the corresponding elements.
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
                int32_t
                channelIdx = elemIdxX % numReceivedChannels;
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
                    int32_t
                    channelIdxShift = elemIdxXShift % numReceivedChannels;
                    LocationType x_elemShift = x_elemsDT[elemIdxXShift];
                    LocationType delayfxElemShift = initialDelay +
                                                    computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elemShift, scanline_x, depth) +
                                                    additionalOffset;
                    int32_t delayElemShift = static_cast<int32_t>(floor(delayfxElemShift));
                    float weightxElemShift = windowFunction->get((x_elemShift - scanline_x) * invMaxElementDistance);
                    float RFxElemShift = RF[delayElemShift + channelIdxShift * numTimesteps +
                                            txScanlineIdx * numReceivedChannels * numTimesteps];
                    float RFxElemShiftedModified = (RFxElemShift < 0) ? ((-1) * std::sqrt(std::abs(RFxElemShift)))
                                                                      : (std::sqrt(RFxElemShift));
                    sample += RFxElemModified * RFxElemShiftedModified;

                    numAcc++;
                    weightAcum += weightxElemShift * weightxElem;
                }
            }
            return sample / numAcc * weightAcum;
        }
    };
}


#endif //SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM3_H
