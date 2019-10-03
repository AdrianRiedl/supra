//
// Created by Adrian Riedl on 2019-10-01.
//

#ifndef SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM_H
#define SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM_H


#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra {

    class RxSampleBeamformerDelayMultiplyAndSum {
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

        /*
         * Similar beamforming as above but changing the x and y axis
         */

        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
        static __device__ ResultType sampleBeamform3DYX(
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

        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
        static __device__ ResultType sampleBeamform3DCombined(
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

        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
        static __device__ ResultType sampleBeamform3DDelayMultiplyAndSum(
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
            int numAdds = 0;
            LocationType initialDelay = txParams.initialDelay;
            uint32_t txScanlineIdx = txParams.txScanlineIdx;
            uint32_t txScanSize = txParams.lastActiveElementIndex.x - txParams.firstActiveElementIndex.x + 1;

            for(int32_t i = 0; i< txScanSize; i++) {
                int32_t channelIdx = (txScanSize + i) % numReceivedChannels;
                LocationType x_elem = x_elemsDT[txParams.firstActiveElementIndex.x + i];
                int32_t sign = 1;
                if (abs(x_elem - scanline_x) <= aDT) {
                    float weight = windowFunction->get((x_elem - scanline_x) * invMaxElementDistance);
                    weightAcum += weight;
                    numAdds++;
                    if (interpolateRFlines) {
                        LocationType delayf = initialDelay +
                                              computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth) +
                                              additionalOffset;
                        int32_t
                        delay = static_cast<int32_t>(floor(delayf));
                        delayf -= delay;
                        if (delay < (numTimesteps - 1)) {
                            uint32_t position = delay + channelIdx * numTimesteps + txScanlineIdx * numReceivedChannels * numTimesteps;
                            float newValue = RF[position + i] * RF[position + ((i+1) % txScanSize)];
                            if(newValue < 0) {
                                sign = -1;
                            } else {
                                sign = 1;
                            }
                            newValue = std::abs(newValue);
                            newValue = std::sqrt(newValue);
                            sample += (sign * newValue);
                        } else if (delay < numTimesteps && delayf == 0.0) {
                            uint32_t position = delay + channelIdx * numTimesteps + txScanlineIdx * numReceivedChannels * numTimesteps;
                            float newValue = RF[position + i] * RF[position + ((i+1) % txScanSize)];
                            if(newValue < 0) {
                                sign = -1;
                                newValue *= (-1);
                            } else {
                                sign = 1;
                            }
                            newValue = std::sqrt(newValue);
                            sample += (sign * newValue);
                        }
                    } else {
                        int32_t
                        delay = static_cast<int32_t>(round(
                                initialDelay + computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth)) +
                                                     additionalOffset);
                        if (delay < numTimesteps) {
                            uint32_t position = delay + channelIdx * numTimesteps + txScanlineIdx * numReceivedChannels * numTimesteps;
                            float newValue = RF[position + i] * RF[position + ((i+1) % txScanSize)];
                            if(newValue < 0) {
                                sign = -1;
                                newValue *= (-1);
                            } else {
                                sign = 1;
                            }
                            newValue = std::sqrt(newValue);
                            sample += (sign * newValue);
                        }
                    }
                }
            }
            if (numAdds > 0) {
                return sample;// / weightAcum * numAdds;
            } else {
                return 0;
            }
        }
    };
}


#endif //SUPRA_RXSAMPLEBEAMFORMERDELAYMULTIPLYANDSUM_H
