// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Adrian Riedl
//      Email Adrian.Riedl@in.tum.de
// 
// ================================================================================================

#ifndef __RXSAMPLEBEAMFORMERDELAYANDSUMXYYX_H__
#define __RXSAMPLEBEAMFORMERDELAYANDSUMXYYX_H__

#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra {

class RxSampleBeamformerDelayAndSumXYYX {
    public:
    /*!
     *  In this beamforming function, the beamforming is done in the X axis, iterating over the Y axis.
     *  It performs the DAS beamforming with weighted values.
     */
    template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
    static __device__ ResultType sampleBeamform3DXY(
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
        ) {
        float sample = 0.0f;
        float weightAcum = 0.0f;
        int numAdds = 0;
        LocationType initialDelay = txParams.initialDelay;
        uint32_t txScanlineIdx = txParams.txScanlineIdx;
        for (uint32_t elemIdxX = txParams.firstActiveElementIndex.x;
             elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++) {
            for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y;
                 elemIdxY < txParams.lastActiveElementIndex.y; elemIdxY++) {
                uint32_t elemIdx = elemIdxX + elemIdxY * elementLayout.x;
                uint32_t channelIdx = elemIdx % numReceivedChannels;
                LocationType x_elem = x_elemsDTsh[elemIdx];
                LocationType z_elem = z_elemsDTsh[elemIdx];

                // squ calculates the square
                if ((squ(x_elem - scanline_x) + squ(z_elem - scanline_z)) <= aDT) {
                    vec2f elementScanlineDistance = {x_elem - scanline_x, z_elem - scanline_z};
                    float weight = computeWindow3DShared(*windowFunction, functionShared,
                                                         elementScanlineDistance * invMaxElementDistance);
                    weightAcum += weight;
                    numAdds++;
                    if (interpolateRFlines) {
                        LocationType delayf = initialDelay +
                                              computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x,
                                                                      scanline_z, depth) + additionalOffset;
                        uint32_t delay = static_cast<uint32_t>(::floor(delayf));
                        delayf -= delay;
                        if (delay < (numTimesteps - 1)) {
                            sample += weight * ((1.0f - delayf) * RF[delay + channelIdx * numTimesteps +
                                                                     txScanlineIdx * numReceivedChannels *
                                                                     numTimesteps] + delayf * RF[(delay + 1) +
                                                                                                 channelIdx *
                                                                                                 numTimesteps +
                                                                                                 txScanlineIdx *
                                                                                                 numReceivedChannels *
                                                                                                 numTimesteps]);
                        } else if (delay < numTimesteps && delayf == 0.0) {
                            sample += weight * RF[delay + channelIdx * numTimesteps +
                                                  txScanlineIdx * numReceivedChannels * numTimesteps];
                        }
                    } else {
                        uint32_t delay = static_cast<uint32_t>(::round(initialDelay +
                                                                       computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem,
                                                                                               z_elem, scanline_x,
                                                                                               scanline_z, depth)) +
                                                               additionalOffset);
                        if (delay < numTimesteps) {
                            sample += weight * RF[delay + channelIdx * numTimesteps +
                                                  txScanlineIdx * numReceivedChannels * numTimesteps];
                        }
                    }
                }
            }
        }

        if (numAdds > 0) {
            return sample / weightAcum * numAdds;
        } else {
            return 0;
        }
    }

    /*!
     *  In this beamforming function, the beamforming is done in the Y axis, iterating over the y axis.
     *  It just switches the X and Y direction of the beamforming function of Class RxSampleBeamformerDelayAndSum
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
        ) {
        float sample = 0.0f;
        float weightAcum = 0.0f;
        int numAdds = 0;
        LocationType initialDelay = txParams.initialDelay;
        uint32_t txScanlineIdx = txParams.txScanlineIdx;
        for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y;
             elemIdxY < txParams.lastActiveElementIndex.y; elemIdxY++) {
            for (uint32_t elemIdxX = txParams.firstActiveElementIndex.x;
                 elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++) {
                uint32_t elemIdx = elemIdxX + elemIdxY * elementLayout.x;
                uint32_t channelIdx = elemIdx % numReceivedChannels;
                LocationType x_elem = x_elemsDTsh[elemIdx];
                LocationType z_elem = z_elemsDTsh[elemIdx];

                // squ calculates the square
                if ((squ(x_elem - scanline_x) + squ(z_elem - scanline_z)) <= aDT) {
                    vec2f elementScanlineDistance = {x_elem - scanline_x, z_elem - scanline_z};
                    float weight = computeWindow3DShared(*windowFunction, functionShared,
                                                         elementScanlineDistance * invMaxElementDistance);
                    weightAcum += weight;
                    numAdds++;
                    if (interpolateRFlines) {
                        LocationType delayf = initialDelay +
                                              computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x,
                                                                      scanline_z, depth) + additionalOffset;
                        uint32_t delay = static_cast<uint32_t>(::floor(delayf));
                        delayf -= delay;
                        if (delay < (numTimesteps - 1)) {
                            sample += weight * ((1.0f - delayf) * RF[delay + channelIdx * numTimesteps +
                                                                     txScanlineIdx * numReceivedChannels *
                                                                     numTimesteps] + delayf * RF[(delay + 1) +
                                                                                                 channelIdx *
                                                                                                 numTimesteps +
                                                                                                 txScanlineIdx *
                                                                                                 numReceivedChannels *
                                                                                                 numTimesteps]);
                        } else if (delay < numTimesteps && delayf == 0.0) {
                            sample += weight * RF[delay + channelIdx * numTimesteps +
                                                  txScanlineIdx * numReceivedChannels * numTimesteps];
                        }
                    } else {
                        uint32_t delay = static_cast<uint32_t>(::round(initialDelay +
                                                                       computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem,
                                                                                               z_elem, scanline_x,
                                                                                               scanline_z, depth)) +
                                                               additionalOffset);
                        if (delay < numTimesteps) {
                            sample += weight * RF[delay + channelIdx * numTimesteps +
                                                  txScanlineIdx * numReceivedChannels * numTimesteps];
                        }
                    }
                }
            }
        }

        if (numAdds > 0) {
            return sample / weightAcum * numAdds;
        } else {
            return 0;
        }
    }

    /*!
     *  In this beamforming function, both results of sampleBeamform3DXY and sampleBeamform3DYX are summed up and
     *  returned.
     */
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
        ResultType beamformedXY = sampleBeamform3DXY<interpolateRFlines, RFType, ResultType, LocationType>(txParams, RF,
                                                                                                         elementLayout,
                                                                                                         numReceivedChannels,
                                                                                                         numTimesteps,
                                                                                                         x_elemsDTsh,
                                                                                                         z_elemsDTsh,
                                                                                                         scanline_x,
                                                                                                         scanline_z,
                                                                                                         dirX, dirY,
                                                                                                         dirZ, aDT,
                                                                                                         depth,
                                                                                                         invMaxElementDistance,
                                                                                                         speedOfSound,
                                                                                                         dt,
                                                                                                         additionalOffset,
                                                                                                         windowFunction,
                                                                                                         functionShared);

        ResultType beamformedYX = sampleBeamform3DYX<interpolateRFlines, RFType, ResultType, LocationType>(txParams, RF,
                                                                                                           elementLayout,
                                                                                                           numReceivedChannels,
                                                                                                           numTimesteps,
                                                                                                           x_elemsDTsh,
                                                                                                           z_elemsDTsh,
                                                                                                           scanline_x,
                                                                                                           scanline_z,
                                                                                                           dirX, dirY,
                                                                                                           dirZ, aDT,
                                                                                                           depth,
                                                                                                           invMaxElementDistance,
                                                                                                           speedOfSound,
                                                                                                           dt,
                                                                                                           additionalOffset,
                                                                                                           windowFunction,
                                                                                                           functionShared);

        return beamformedXY + beamformedYX;
    }

    /*!
     * Will be called by the system, syntax must not be changed.
     * This function is used to call the corresponding function to perform the beamforming on the raw data.
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
    ) {
        return sampleBeamform3DCombined<interpolateRFlines, RFType, ResultType, LocationType>(txParams, RF, elementLayout, numReceivedChannels, numTimesteps, x_elemsDTsh,
                                        z_elemsDTsh, scanline_x, scanline_z, dirX, dirY, dirZ, aDT, depth,
                                        invMaxElementDistance, speedOfSound, dt, additionalOffset, windowFunction,
                                        functionShared);
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

        for (int32_t elemIdxX = txParams.firstActiveElementIndex.x;
             elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++) {
            int32_t channelIdx = elemIdxX % numReceivedChannels;
            LocationType x_elem = x_elemsDT[elemIdxX];
            if (abs(x_elem - scanline_x) <= aDT) {
                float weight = windowFunction->get((x_elem - scanline_x) * invMaxElementDistance);
                weightAcum += weight;
                numAdds++;
                if (interpolateRFlines) {
                    LocationType delayf = initialDelay +
                                          computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth) +
                                          additionalOffset;
                    int32_t delay = static_cast<int32_t>(floor(delayf));
                    delayf -= delay;
                    if (delay < (numTimesteps - 1)) {
                        sample += weight * ((1.0f - delayf) * RF[delay + channelIdx * numTimesteps +
                                                                 txScanlineIdx * numReceivedChannels * numTimesteps] +
                                            delayf * RF[(delay + 1) + channelIdx * numTimesteps +
                                                        txScanlineIdx * numReceivedChannels * numTimesteps]);
                    } else if (delay < numTimesteps && delayf == 0.0) {
                        sample += weight * RF[delay + channelIdx * numTimesteps +
                                              txScanlineIdx * numReceivedChannels * numTimesteps];
                    }
                } else {
                    int32_t delay = static_cast<int32_t>(
                            round(initialDelay + computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth)) +
                            additionalOffset);
                    if (delay < numTimesteps) {
                        sample += weight * RF[delay + channelIdx * numTimesteps +
                                              txScanlineIdx * numReceivedChannels * numTimesteps];
                    }
                }
            }
        }
        if (numAdds > 0) {
            return sample / weightAcum * numAdds;
        } else {
            return 0;
        }
    }
};
}

#endif //!__RXSAMPLEBEAMFORMERDELAYANDSUMXYYX_H__
