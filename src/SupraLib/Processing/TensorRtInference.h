// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __TENSORRTINFERENCE_H__
#define __TENSORRTINFERENCE_H__

#ifdef HAVE_TENSORRT

#include <memory>

#include <Container.h>
#include <vec.h>
#include <utilities/Logging.h>

namespace nvinfer1
{
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace supra
{
	class TensorRtInference {
	public:
		TensorRtInference(
			const std::string& modelFilename, 
			const std::string& inputNormalization, 
			const std::string& outputDenormalization);

		~TensorRtInference();

        std::shared_ptr<ContainerBase> process(
                std::shared_ptr<const Container<float> >imageData,
                vec3s inputSize, vec3s outputSize,
                const std::string& currentLayout, const std::string& finalLayout,
                DataType modelInputDataType, DataType modelOutputDataType,
                const std::string& modelInputLayout, const std::string& modelOutputLayout,
                uint32_t inferencePatchSize, uint32_t inferencePatchOverlap);

		/*template <typename ModelOutputType, typename OutputType>
		void copyPatchToOutput(
		        const torch::Tensor& output,
		        std::shared_ptr<Container<OutputType> > pDataOut,
		        const std::string& modelOutputLayout,
		        const std::string& finalLayout,
		        vec3s outputSize, size_t startPixelValid, size_t startPixel, size_t patchSizeValid)
        {
            // Copy to the result buffer (while converting to OutputType)
            auto outAccessor = output.accessor<ModelOutputType, 4>();

            // Determine which output dimension is affected by the patching
            auto permutation = layoutPermutation(modelOutputLayout, finalLayout);

            size_t outSliceStart = 0;
            size_t outSliceEnd = outputSize.z;
            size_t outSliceOffset = 0;
            size_t outLineStart = 0;
            size_t outLineEnd = outputSize.y;
            size_t outLineOffset = 0;
            size_t outPixelStart = 0;
            size_t outPixelEnd = outputSize.x;
            size_t outPixelOffset = 0;

            // Since the data is already permuted to the right layout for output, but we are working patchwise,
            // we need to restrict the affected dimension's indices
            if (permutation[1] == 3)
            {
                outSliceStart = startPixelValid;
                outSliceEnd = startPixelValid + patchSizeValid;
                outSliceOffset = startPixel;
            }
            else if (permutation[2] == 3)
            {
                outLineStart = startPixelValid;
                outLineEnd = startPixelValid + patchSizeValid;
                outLineOffset = startPixel;
            }
            else if (permutation[3] == 3)
            {
                outPixelStart = startPixelValid;
                outPixelEnd = startPixelValid + patchSizeValid;
                outPixelOffset = startPixel;
            }

            for (size_t outSliceIdx = outSliceStart; outSliceIdx < outSliceEnd; outSliceIdx++)
            {
                for (size_t outLineIdx = outLineStart; outLineIdx < outLineEnd; outLineIdx++)
                {
                    for (size_t outPixelIdx = outPixelStart; outPixelIdx < outPixelEnd; outPixelIdx++)
                    {
                        pDataOut->get()[
                                outSliceIdx * outputSize.y * outputSize.x +
                                outLineIdx * outputSize.x +
                                outPixelIdx] =
                                clampCast<OutputType>(outAccessor[0]
                                                      [outSliceIdx - outSliceOffset]
                                                      [outLineIdx - outLineOffset]
                                                      [outPixelIdx - outPixelOffset]);
                    }
                }
            }
        }*/

	private:
		void loadModel();
		void unloadModel();
		//at::Tensor convertDataType(at::Tensor tensor, DataType datatype);
		//at::Tensor changeLayout(at::Tensor tensor, const std::string& currentLayout, const std::string& outLayout);
		std::vector<int64_t> layoutPermutation(const std::string& currentLayout, const std::string& outLayout);

		//std::shared_ptr<torch::jit::script::Module> m_torchModule;
		//std::shared_ptr<torch::jit::script::Module> m_inputNormalizationModule;
		//std::shared_ptr<torch::jit::script::Module> m_outputDenormalizationModule;

        nvinfer1::IRuntime* m_tensorRtRuntime;
        nvinfer1::ICudaEngine* m_tensorRtEngine;
        nvinfer1::IExecutionContext* m_tensorRtContext;


		std::string m_modelFilename;
		std::string m_inputNormalization;
		std::string m_outputDenormalization;
	};
}

#endif //HAVE_TENSORRT

#endif //!__TENSORRTINFERENCE_H__
