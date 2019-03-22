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

#include "TensorRtInference.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
#include "NvInferPlugin.h"

using namespace std;

namespace supra
{
    // Logger for TensorRT info/warning/errors
    class Logger : public nvinfer1::ILogger
    {
    public:
        Logger(Severity severity = Severity::kINFO)
                : reportableSeverity(severity)
        {
        }

        void log(Severity severity, const char* msg) override
        {
            // suppress messages with severity enum value greater than the reportable
            if (severity > reportableSeverity)
                return;

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR:
                    logging::log_error("TensorRT INTERNAL_ERROR: ", msg);
                    break;
                case Severity::kERROR:
                    logging::log_error("TensorRT: ", msg);
                    break;
                case Severity::kWARNING:
                    logging::log_warn("TensorRT: ", msg);
                    break;
                case Severity::kINFO:
                    logging::log_info("TensorRT: ", msg);
                    break;
                default:
                    logging::log_error("TensorRT: ", msg);
                    break;
            }
        }

        Severity reportableSeverity;
    };

    static Logger gLogger;
    static int gUseDLACore = -1;//samplesCommon::parseDLA(argc, argv);
    static const int INPUT_H = 28;
    static const int INPUT_W = 28;
    static const int OUTPUT_SIZE = 10;

    inline void enableDLA(nvinfer1::IBuilder* b, int useDLACore)
    {
        if (useDLACore >= 0)
        {
            b->allowGPUFallback(true);
            b->setFp16Mode(true);
            b->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            b->setDLACore(useDLACore);
        }
    }

    void onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                        unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                        nvinfer1::IHostMemory*& trtModelStream) // output buffer for the TensorRT model
    {
        int verbosity = (int) nvinfer1::ILogger::Severity::kINFO;

        // create the builder
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();

        auto parser = nvonnxparser::createParser(*network, gLogger);

        //Optional - uncomment below lines to view network layer information
        //config->setPrintLayerInfo(true);
        //parser->reportParsingInfo();

        if (!parser->parseFromFile(modelFile.c_str(), verbosity))
        {
            string msg("failed to parse onnx file");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());

            for (int k = 0; k < parser->getNbErrors(); k++)
            {
                auto err = parser->getError(k);
                logging::log_error("TensorRT Parser error: (", k, ")  code ", (int)err->code());
                logging::log_error("TensorRT Parser error: (", k, ") ", err->desc());
                logging::log_error("TensorRT Parser error: (", k, ") ", err->file(), ", ", err->line(), ", ", err->func());
            }
            exit(EXIT_FAILURE);
        }

        // Build the engine
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 20);

        enableDLA(builder, gUseDLACore);
        nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
        assert(engine);

        // we can destroy the parser
        parser->destroy();

        // serialize the engine, then close everything down
        trtModelStream = engine->serialize();
        engine->destroy();
        network->destroy();
        builder->destroy();
    }

    void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize)
    {
        const nvinfer1::ICudaEngine& engine = context.getEngine();
        // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        int inputIndex, outputIndex;
        for (int b = 0; b < engine.getNbBindings(); ++b)
        {
            if (engine.bindingIsInput(b))
                inputIndex = b;
            else
                outputIndex = b;
        }

        // create GPU buffers and a stream
        cudaSafeCall(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
        cudaSafeCall(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        cudaStream_t stream;
        cudaSafeCall(cudaStreamCreate(&stream));

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        cudaSafeCall(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        cudaSafeCall(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // release the stream and the buffers
        cudaStreamDestroy(stream);
        cudaSafeCall(cudaFree(buffers[inputIndex]));
        cudaSafeCall(cudaFree(buffers[outputIndex]));
    }

    TensorRtInference::TensorRtInference(
		const std::string& modelFilename,
		const std::string& inputNormalization,
		const std::string& outputDenormalization)
		: m_modelFilename{ modelFilename }
		, m_inputNormalization{inputNormalization}
		, m_outputDenormalization{outputDenormalization}
		, m_tensorRtRuntime{ nullptr }
		, m_tensorRtEngine{ nullptr }
		, m_tensorRtContext{ nullptr }
	{
		if (m_inputNormalization == "")
		{
			m_inputNormalization = "a";
		}
		if (m_outputDenormalization == "")
		{
			m_outputDenormalization = "a";
		}

        loadModel();
	}

	TensorRtInference::~TensorRtInference()
    {
        unloadModel();
    }

	void TensorRtInference::loadModel() {
        /*nvonnxparser::IOnnxConfig* config = nvonnxparser::createONNXConfig();
//Create Parser
        nvonnxparser::IONNXParser* parser = nvonnxparser::createONNXParser(*config);
        parser->parse(onnx_filename, DataType::kFLOAT);
        parser->convertToTRTNetwork();
        nvinfer1::INetworkDefinition* trtNetwork = parser->getTRTNetwork();




        IExecutionContext *context = engine->createExecutionContext();

        int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
        int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        void* buffers[2];
        buffers[inputIndex] = inputbuffer;
        buffers[outputIndex] = outputBuffer;

        context.enqueue(batchSize, buffers, stream, nullptr);*/

        unloadModel();

        // create a TensorRT model from the onnx model and serialize it to a stream
        nvinfer1::IHostMemory* trtModelStream{nullptr};
        onnxToTRTModel(m_modelFilename, 1, trtModelStream);
        assert(trtModelStream != nullptr);

        // read a random digit file
        srand(unsigned(time(nullptr)));
        uint8_t fileData[INPUT_H * INPUT_W];
        int num = rand() % 10;
        //readPGMFile(std::to_string(num) + ".pgm", fileData);

        float data[INPUT_H * INPUT_W];
        for (int i = 0; i < INPUT_H * INPUT_W; i++)
            data[i] = 1.0 - float(fileData[i] / 255.0);

        // deserialize the engine
        m_tensorRtRuntime = nvinfer1::createInferRuntime(gLogger);
        assert(m_tensorRtRuntime != nullptr);
        if (gUseDLACore >= 0)
        {
            m_tensorRtRuntime->setDLACore(gUseDLACore);
        }

        m_tensorRtEngine = m_tensorRtRuntime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
        assert(m_tensorRtEngine != nullptr);
        trtModelStream->destroy();
        m_tensorRtContext = m_tensorRtEngine->createExecutionContext();
        assert(m_tensorRtContext != nullptr);


		/*m_torchModule = nullptr;
		m_inputNormalizationModule = nullptr;
		m_outputDenormalizationModule = nullptr;
		logging::log_error_if(m_modelFilename == "",
				"TensorRtInference: Error while loading model: Model path is empty.");
		logging::log_error_if(m_inputNormalization == "",
				"TensorRtInference: Error while building module: Normalization string is empty.");
		logging::log_error_if(m_outputDenormalization == "",
				"TensorRtInference: Error while building module: Denormalization string is empty.");
		if (m_modelFilename != "")
		{
		    if (fileExists(m_modelFilename))
            {
                try {
                    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(m_modelFilename);
                    module->to(torch::kCUDA);
                    m_torchModule = module;
                }
                catch (c10::Error e)
                {
                    logging::log_error("TensorRtInference: Exception (c10::Error) while loading model '", m_modelFilename, "'");
                    logging::log_error("TensorRtInference: ", e.what());
                    logging::log_error("TensorRtInference: ", e.msg_stack());
                    m_torchModule = nullptr;
                }
                catch (std::runtime_error e)
                {
                    logging::log_error("TensorRtInference: Exception (std::runtime_error) while loading model '", m_modelFilename, "'");
                    logging::log_error("TensorRtInference: ", e.what());
                    m_torchModule = nullptr;
                }
			}
		    else
            {
                logging::log_error("TensorRtInference: Error while loading model '", m_modelFilename, "'. The file does not exist.");
            }
		}
		if (m_inputNormalization != "")
		{
			try {
				m_inputNormalizationModule = torch::jit::compile(
						"  def normalize(a):\n    return " + m_inputNormalization + "\n");
				m_inputNormalizationModule->to(torch::kCUDA);
			}
			catch (c10::Error e)
			{
				logging::log_error("TensorRtInference: Exception (c10::Error) while building normalization module '", m_inputNormalization, "'");
				logging::log_error("TensorRtInference: ", e.what());
				logging::log_error("TensorRtInference: ", e.msg_stack());
				m_inputNormalizationModule = nullptr;
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TensorRtInference: Exception (std::runtime_error) while building normalization module '", m_inputNormalization, "'");
				logging::log_error("TensorRtInference: ", e.what());
				m_inputNormalizationModule = nullptr;
			}
		}
		if (m_outputDenormalization != "")
		{
			try {
				m_outputDenormalizationModule = torch::jit::compile(
						"  def denormalize(a):\n    return " + m_outputDenormalization + "\n");
				m_outputDenormalizationModule->to(torch::kCUDA);
			}
			catch (c10::Error e)
			{
				logging::log_error("TensorRtInference: Exception (c10::Error) while building denormalization module '", m_outputDenormalization, "'");
				logging::log_error("TensorRtInference: ", e.what());
				logging::log_error("TensorRtInference: ", e.msg_stack());
				m_outputDenormalizationModule = nullptr;
			}
			catch (std::runtime_error e)
			{
				logging::log_error("TensorRtInference: Exception (std::runtime_error) while building denormalization module '", m_outputDenormalization, "'");
				logging::log_error("TensorRtInference: ", e.what());
				m_outputDenormalizationModule = nullptr;
			}
		}*/
	}

	void TensorRtInference::unloadModel()
    {
        // destroy the engine
        if (m_tensorRtContext)
        {
            m_tensorRtContext->destroy();
            m_tensorRtContext = nullptr;
        }
        if (m_tensorRtEngine)
        {
            m_tensorRtEngine->destroy();
            m_tensorRtEngine = nullptr;
        }
        if (m_tensorRtRuntime)
        {
            m_tensorRtRuntime->destroy();
            m_tensorRtRuntime = nullptr;
        }
    }

    //template <typename InputType, typename OutputType>
    using InputType = float;
    using OutputType = float;
    std::shared_ptr<ContainerBase> TensorRtInference::process(
            std::shared_ptr<const Container<InputType> >imageData,
            vec3s inputSize, vec3s outputSize,
            const std::string& currentLayout, const std::string& finalLayout,
            DataType modelInputDataType, DataType modelOutputDataType,
            const std::string& modelInputLayout, const std::string& modelOutputLayout,
            uint32_t inferencePatchSize, uint32_t inferencePatchOverlap)
    {
        std::shared_ptr <Container<OutputType> > pDataOut = nullptr;

        if (m_tensorRtContext != nullptr)
        {
            try {
                // Synchronize to the stream the input data is produced on, to make sure the data is ready
                //cudaSafeCall(cudaStreamSynchronize(imageData->getStream()));

                // Wrap our input data into a torch tensor (with first dimension 1, for batchsize 1)
                /*auto inputData = torch::from_blob((void*)(imageData->get()),
                                                  { (int64_t)1, (int64_t)inputSize.z, (int64_t)inputSize.y, (int64_t)inputSize.x },
                                                  torch::TensorOptions().
                                                          dtype(caffe2::TypeMeta::Make<InputType>()).
                                                          device(imageData->isGPU() ? torch::kCUDA : torch::kCPU).
                                                          requires_grad(false));

                if (inferencePatchSize == 0)
                {
                    inferencePatchSize = inputSize.x;
                }
                assert(inferencePatchSize > inferencePatchOverlap * 2);

                size_t numPixels = inputSize.x;*/
                pDataOut = std::make_shared<Container<OutputType> >(LocationHost, imageData->getStream(), outputSize.x*outputSize.y*outputSize.z);


                // run inference
                assert(m_tensorRtEngine->getNbBindings() == 2);
                void* buffers[2];

                // In order to bind the buffers, we need to know the names of the input and output tensors.
                // note that indices are guaranteed to be less than IEngine::getNbBindings()
                int inputIndex, outputIndex;
                for (int b = 0; b < m_tensorRtEngine->getNbBindings(); ++b)
                {
                    if (m_tensorRtEngine->bindingIsInput(b))
                        inputIndex = b;
                    else
                        outputIndex = b;
                }
                buffers[inputIndex] = (void*)(imageData->get());
                buffers[outputIndex] = pDataOut->get();

                m_tensorRtContext->enqueue(1, buffers, imageData->getStream(), nullptr);
                cudaSafeCall(cudaPeekAtLastError());


                /*doInference(*m_tensorRtContext, imageData->get(), pDataOut->get(), 1);

                size_t lastValidPixels = 0;
                for (size_t startPixelValid = 0; startPixelValid < numPixels; startPixelValid += lastValidPixels)
                {
                    // Compute the size and position of the patch we want to run the model for
                    size_t patchSizeValid = 0;
                    size_t patchSize = 0;
                    size_t startPixel = 0;
                    if (startPixelValid == 0 && numPixels - startPixelValid <= inferencePatchSize)
                    {
                        //Special case: The requested patch size is large enough. No patching necessary!
                        patchSize = numPixels - startPixelValid;
                        patchSizeValid = patchSize;
                        startPixel = 0;
                    }
                    else if (startPixelValid == 0)
                    {
                        // The first patch only needs to be padded on the bottom
                        patchSize = inferencePatchSize;
                        patchSizeValid = patchSize - inferencePatchOverlap;
                        startPixel = 0;
                    }
                    else if (numPixels - (startPixelValid - inferencePatchOverlap) <= inferencePatchSize)
                    {
                        // The last patch only needs to be padded on the top
                        startPixel = (startPixelValid - inferencePatchOverlap);
                        patchSize = numPixels - startPixel;
                        patchSizeValid = patchSize - inferencePatchOverlap;
                    }
                    else
                    {
                        // Every patch in the middle
                        // padding on the top and bottom
                        startPixel = (startPixelValid - inferencePatchOverlap);
                        patchSize = inferencePatchSize;
                        patchSizeValid = patchSize - 2 * inferencePatchOverlap;
                    }
                    lastValidPixels = patchSizeValid;

                    // Slice the input data
                    auto inputDataPatch = inputData.slice(3, startPixel, startPixel + patchSize);

                    // Convert it to the desired input type
                    inputDataPatch = convertDataType(inputDataPatch, modelInputDataType);

                    // Adjust layout if necessary
                    inputDataPatch = changeLayout(inputDataPatch, currentLayout, modelInputLayout);
                    assert(!(inputDataPatch.requires_grad()));

                    // Run model
                    // Normalize the input
                    auto inputDataPatchIvalue = m_inputNormalizationModule->run_method("normalize", inputDataPatch);
                    cudaSafeCall(cudaPeekAtLastError());

                    // build module input data structure
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(inputDataPatchIvalue);

                    // Execute the model and turn its output into a tensor.
                    auto result = m_torchModule->forward(inputs);
                    cudaSafeCall(cudaPeekAtLastError());

                    // Denormalize the output
                    result = m_outputDenormalizationModule->run_method("denormalize", result);
                    cudaSafeCall(cudaPeekAtLastError());
                    at::Tensor output = result.toTensor();
                    // This should never happen right now.
                    assert(!output.is_hip());

                    // Adjust layout
                    output = changeLayout(output, modelOutputLayout, finalLayout);
                    if (modelOutputDataType == TypeHalf)
                    {
                        // As half is not natively supported on the CPU, promote it to float
                        output = output.to(torch::kFloat);
                    }
                    output = output.to(torch::kCPU);

                    // Copy the patch!
                    if (output.dtype() == torch::kInt8)
                    {
                        copyPatchToOutput<int8_t, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                              outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                    else if (output.dtype() == torch::kUInt8)
                    {
                        copyPatchToOutput<uint8_t, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                               outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                    else if (output.dtype() == torch::kInt16)
                    {
                        copyPatchToOutput<int16_t, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                               outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                    else if (output.dtype() == torch::kInt32)
                    {
                        copyPatchToOutput<int32_t, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                               outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                    else if (output.dtype() == torch::kInt64)
                    {
                        copyPatchToOutput<int64_t, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                               outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                    else if (output.dtype() == torch::kFloat)
                    {
                        copyPatchToOutput<float, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                             outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                    else if(output.dtype() == torch::kDouble)
                    {
                        copyPatchToOutput<double, OutputType>(output, pDataOut, modelOutputLayout, finalLayout,
                                                              outputSize, startPixelValid, startPixel, patchSizeValid);
                    }
                }
                cudaSafeCall(cudaPeekAtLastError()); */
            }
            catch (std::runtime_error& e)
            {
                logging::log_error("TensorRtInference: Error (std::runtime_error) while running model '", m_modelFilename, "'");
                logging::log_error("TensorRtInference: ", e.what());
            }
        }
        else
        {
            logging::log_error_if(m_tensorRtContext == nullptr,
                                  "TensorRtInference: Error no model loaded.");
            /*logging::log_error_if(m_inputNormalizationModule == nullptr,
                                  "TensorRtInference: Error no normalization module present.");
            logging::log_error_if(m_outputDenormalizationModule == nullptr,
                                  "TensorRtInference: Error no denormalization module present.");*/
        }

        return pDataOut;
    }

	/*at::Tensor supra::TensorRtInference::convertDataType(at::Tensor tensor, DataType datatype)
	{
		switch (datatype)
		{
		case TypeInt8:
			tensor = tensor.to(caffe2::TypeMeta::Make<int8_t>());
			break;
		case TypeUint8:
			tensor = tensor.to(caffe2::TypeMeta::Make<uint8_t>());
			break;
		case TypeInt16:
			tensor = tensor.to(caffe2::TypeMeta::Make<int16_t>());
			break;
		case TypeInt32:
			tensor = tensor.to(caffe2::TypeMeta::Make<int32_t>());
			break;
		case TypeInt64:
			tensor = tensor.to(caffe2::TypeMeta::Make<int64_t>());
			break;
		case TypeHalf:
			tensor = tensor.to(at::kHalf);
			break;
		case TypeFloat:
			tensor = tensor.to(caffe2::TypeMeta::Make<float>());
			break;
		case TypeDouble:
			tensor = tensor.to(caffe2::TypeMeta::Make<double>());
			break;
		default:
			logging::log_error("TensorRtInference: convertDataType: Type '", datatype, "' is not supported.");
			break;
		}
		return tensor;
	}*/

	/*at::Tensor supra::TensorRtInference::changeLayout(at::Tensor tensor, const std::string & currentLayout, const std::string & outLayout)
	{
		if (currentLayout != outLayout)
		{
			auto permutation = layoutPermutation(currentLayout, outLayout);
			tensor = tensor.permute(permutation);
		}
		return tensor;
	}*/

	std::vector<int64_t> TensorRtInference::layoutPermutation(const std::string& currentLayout, const std::string& outLayout)
	{
		int inDimensionC = (currentLayout[0] == 'C' ? 1 : (currentLayout[2] == 'C' ? 2 : 3));
		int inDimensionW = (currentLayout[0] == 'W' ? 1 : (currentLayout[2] == 'W' ? 2 : 3));
		int inDimensionH = (currentLayout[0] == 'H' ? 1 : (currentLayout[2] == 'H' ? 2 : 3));
		int outDimensionC = (outLayout[0] == 'C' ? 1 : (outLayout[2] == 'C' ? 2 : 3));
		int outDimensionW = (outLayout[0] == 'W' ? 1 : (outLayout[2] == 'W' ? 2 : 3));
		int outDimensionH = (outLayout[0] == 'H' ? 1 : (outLayout[2] == 'H' ? 2 : 3));

		std::vector<int64_t> permutation(4, 0);
		permutation[0] = 0;
		permutation[outDimensionC] = inDimensionC;
		permutation[outDimensionW] = inDimensionW;
		permutation[outDimensionH] = inDimensionH;

		return permutation;
	}
}
