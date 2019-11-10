//
// Created by Adrian Riedl on 2019-10-10.
//

#ifndef SUPRA_RXSAMPLEBEAMFORMERSIGNEDDELAYMULTIPLYANDSUM_H
#define SUPRA_RXSAMPLEBEAMFORMERSIGNEDDELAYMULTIPLYANDSUM_H

#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra {

    class RxSampleBeamformerSignedDelayMultiplyAndSum {
    public:
        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType> static __device__ ResultType
        sampleBeamform3D(
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
         * Implementation of the signed Delay Multiply and Sum algorithm according to the paper
         * 'Signed Real-Time Delay Multiply and Sum Beamforming for Multispectral Photoacoustic Imaging' (source: https://www.researchgate.net/publication/328335303_Signed_Real-Time_Delay_Multiply_and_Sum_Beamforming_for_Multispectral_Photoacoustic_Imaging)
         *
         * Using the implemented beamforming function in RxSampleBeamformerDelayAndSum.h and RxSampleBeamformerDelayMultiplyAndSum.h.
         *
         *
         * @tparam interpolateRFlines
         * @tparam RFType
         * @tparam ResultType
         * @tparam LocationType
         * @param txParams
         * @param RF
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
         * @param speedOfSound
         * @param dt
         * @param additionalOffset
         * @param windowFunction
         * @return                                  The value of the beamformed raw data.
         */
        template<bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType> static __device__ ResultType
        sampleBeamform2D(
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
            ResultType sign = (RxSampleBeamformerDelayAndSum::sampleBeamform2D
                    <interpolateRFlines, RFType, ResultType, LocationType>(
                    txParams,
                    RF,
                    numTransducerElements,
                    numReceivedChannels,
                    numTimesteps,
                    x_elemsDT,
                    scanline_x,
                    dirX,
                    dirY,
                    dirZ,
                    aDT,
                    depth,
                    invMaxElementDistance,
                    speedOfSound,
                    dt,
                    additionalOffset,
                    windowFunction));

            ResultType dmasValue = RxSampleBeamformerDelayMultiplyAndSum::sampleBeamform2D
                    <interpolateRFlines, RFType, ResultType, LocationType>(
                    txParams,
                    RF,
                    numTransducerElements,
                    numReceivedChannels,
                    numTimesteps,
                    x_elemsDT,
                    scanline_x,
                    dirX,
                    dirY,
                    dirZ,
                    aDT,
                    depth,
                    invMaxElementDistance,
                    speedOfSound,
                    dt,
                    additionalOffset,
                    windowFunction);

            return (sign < 0) ? ((-1) * dmasValue) : dmasValue;
        }
    };
}

#endif //SUPRA_RXSAMPLEBEAMFORMERSIGNEDDELAYMULTIPLYANDSUM_H
