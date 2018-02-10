// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "UsIntDemonstrator.h"

#include <memory>

//#include <set>

#include "USImage.h"
#include "Beamformer/Sequencer.h"
#include "Beamformer/USTransducer.h"
#include "utilities/utility.h"
#include "utilities/CallFrequency.h"
#include "utilities/Logging.h"
#ifdef HAVE_WIRINGPI
#include <wiringPi.h>
#endif

namespace supra
{
	UsDemonstrator::UsDemonstrator()
		: m_txClock(1000)
		, m_pulseFrequency(0.5)
		, m_running(false)
		, m_frozen(false)
		, m_currentFrame(0)
		, m_currentTransmit(0)
	{
		m_timer.setFrequency(m_pulseFrequency);
		m_txSleepUs = static_cast<unsigned int>(1000000.0 / m_txClock);
#ifdef HAVE_WIRINGPI
		wiringPiSetup();
		setPinMode(OUTPUT);
		for(const uint8_t& pin : m_pins)
		{
			digitalWrite(pin, false);
		}
#endif
	}

	UsDemonstrator::~UsDemonstrator()
	{
		for(const uint8_t& pin : m_pins)
		{
			digitalWrite(pin, false);
			pinMode(pin, INPUT);
			pullUpDnControl(pin, PUD_DOWN) ;
		}
	}

	void UsDemonstrator::addFrame(const Frame& frame)
	{
		std::lock_guard<std::mutex> lock(m_objMutex);

		size_t maxLength = 0;
		size_t minLength = std::numeric_limits<size_t>::max();
		for (const auto& transmit : frame)
		{
			for (const auto& txWave : transmit.txWaves)
			{
				maxLength = max(maxLength, txWave.size());
				minLength = min(minLength, txWave.size());
			}
		}
		auto frameEq = frame;
		if (maxLength != minLength)
		{
			for (auto& transmit : frameEq)
			{
				for (auto& txWave : transmit.txWaves)
				{
					txWave.resize(maxLength, 0);
				}
			}
		}

		m_frames.push_back(frameEq);
	}

	void UsDemonstrator::executionLoop()
	{
#ifdef HAVE_WIRINGPI
		piHiPri(90);
#endif
		while (m_running)
		{
			if(!m_frozen && m_frames.size() > 0)
			{
				std::lock_guard<std::mutex> lock(m_objMutex);
#ifdef HAVE_WIRINGPI
				// on Raspberry: Pulse!
				transmitPulse(m_frames[m_currentFrame][m_currentTransmit]);
//#else
#endif //HAVE_WIRINGPI
				// publish the pulse image
				if (m_callback)
				{
					m_callback(m_frames[m_currentFrame][m_currentTransmit].txWaves);
				}
//#endif //HAVE_WIRINGPI
				logging::log_info("UsDemonstrator loop. Frame ", m_currentFrame + 1, " / ", m_frames.size(), ", Transmit ",
					m_currentTransmit + 1, " / ", m_frames[m_currentFrame].size());

				m_currentTransmit = (m_currentTransmit + 1) % m_frames[m_currentFrame].size();
				if (m_currentTransmit == 0)
				{
					m_currentFrame = (m_currentFrame + 1) % m_frames.size();
				}
			}
			m_timer.sleepUntilNextSlot();
		}
	}

	void UsDemonstrator::start()
	{
		m_running = true;
	}

	void UsDemonstrator::stop()
	{
		m_running = false;
	}
	void UsDemonstrator::freeze()
	{
		m_frozen = true;
	}
	void UsDemonstrator::unfreeze()
	{
		m_frozen = false;
	}
	void UsDemonstrator::setCallback(std::function<void(const std::vector<std::vector<uint8_t>>& )> callback)
	{
		std::lock_guard<std::mutex> lock(m_objMutex);
		m_callback = callback;
	}

	void UsDemonstrator::setPulseFrequency(double pulseFrequency)
	{
		std::lock_guard<std::mutex> lock(m_objMutex);
		m_pulseFrequency = pulseFrequency;
		m_timer.setFrequency(m_pulseFrequency);
	}

	void UsDemonstrator::setPinMode(int mode)
	{
#ifdef HAVE_WIRINGPI
		for (const uint8_t& pin : m_pins)
		{
			pinMode(pin, mode);
		}
#endif
	}

	void UsDemonstrator::transmitPulse(const TransmitBeam& beam)
	{
#ifdef HAVE_WIRINGPI
		size_t numEntries = beam.txWaves[0].size();
		size_t numElements = beam.txWaves.size();

		unsigned int loopStart = micros();
		for (size_t k = 0; k < numEntries; k++)
		{
			unsigned int microsIn = micros();
			loopStart += m_txSleepUs;
			for(size_t elem = 0; elem < numElements; elem++)
			{
				digitalWrite(m_pins[elem], beam.txWaves[elem][k]);
			}
			delayUntil(loopStart);
		}
#endif
	}

	void UsDemonstrator::delayUntil(unsigned int targetTime)
	{
		while (micros() <= targetTime) {};
	};

	constexpr uint8_t UsDemonstrator::m_pins[USDEMONSTRATOR_NUM_PINS];
}

namespace supra
{
	using namespace std;
	using logging::log_error;
	using logging::log_log;

	UsIntDemonstrator::UsIntDemonstrator(tbb::flow::graph & graph, const std::string& nodeID)
		: AbstractInput<RecordObject>(graph, nodeID)
		, m_pTransducer(nullptr)
		, m_pSequencer(nullptr)
		, m_numMuxedChannels(0)
		, m_probeMapping(0)
		, m_numBeamSequences(0) // TODO replace hardcoded sequence number (move to config/gui)
		, m_interface(nullptr)
//		, m_lastFrameNumber(1)
//		, m_sequenceNumFrames(0)
	{
		m_callFrequency.setName("CepUS");

		m_ready = false;

		// number of different frames (e.g. 1 PW follwed by 1 B-Mode) as basic determinator for other settings
		m_valueRangeDictionary.set<uint32_t>("sequenceNumFrames", 1, 10, 1, "Number of different beam sequences");
		m_numBeamSequences = 1; // not known at startup as this will be updated only once xml configuration is read

		//Setup allowed values for parameters
		m_valueRangeDictionary.set<uint32_t>("systemTxClock", 100, 10000, 1000, "TX system clock (Hz)");
		m_valueRangeDictionary.set<string>("probeName", {"LEDs", "Demonstrator"}, "LEDs", "Probe");

		m_valueRangeDictionary.set<double>("startDepth", 0.0, 300.0, 0.0, "Start depth [mm]");
		m_valueRangeDictionary.set<double>("endDepth", 0.0, 300.0, 70.0, "End depth [mm]");
		m_valueRangeDictionary.set<bool>("measureThroughput", {false, true}, false, "Measure throughput");
		m_valueRangeDictionary.set<double>("speedOfSound", 0.001, 10, 0.284, "Speed of sound [m/s]"); // Default value: for 100 Hz waves
		// with f = 100Hz
		// c = 0.3580 m / s
		// lambda = 0.0036 m

		// set beamSeq specific value ranges, too.
		setBeamSequenceValueRange(0);


		// create new sequencer
		m_pSequencer = unique_ptr<Sequencer>(new Sequencer(m_numBeamSequences));
		m_beamEnsembleTxParameters.resize(m_numBeamSequences);

		configurationChanged();
	}

	UsIntDemonstrator::~UsIntDemonstrator()
	{
		m_interface->stop();
		if (m_interfaceRunnerThread.joinable())
		{
			m_interfaceRunnerThread.join();
		}
		//if (m_cUSEngine)
		//{
			//End of the world, waiting to join completed thread on teardown
		//}
		/*if (m_runEngineThread.joinable())
		{
			m_runEngineThread.join();
		}*/
	}

	// return appendix for configuration strings for a given beam sequence
	// consider backward compatibility for a single beam sequence, where the appendix strings are simply ommited
	std::string UsIntDemonstrator::getBeamSequenceApp(size_t totalSequences, size_t sequenceId)
	{
		if (totalSequences == 1)
		{
			return std::string("");
		}
		else
		{
			return "seq"+std::to_string(sequenceId)+"_";
		}
	}

	void UsIntDemonstrator::setBeamSequenceValueRange(size_t oldBeamSequenceValueRange)
	{
		for (size_t numSeq = 0; numSeq < oldBeamSequenceValueRange; ++numSeq)
		{
				// remove old keys
				std::string idApp = getBeamSequenceApp(oldBeamSequenceValueRange,numSeq);

				m_valueRangeDictionary.remove(idApp+"scanType");
				m_valueRangeDictionary.remove(idApp+"txVoltage");
				m_valueRangeDictionary.remove(idApp+"txFrequency");
				m_valueRangeDictionary.remove(idApp+"txPulseRepetitionFrequency");
				m_valueRangeDictionary.remove(idApp+"txWindowType");
				m_valueRangeDictionary.remove(idApp+"txWindowParameter");
				m_valueRangeDictionary.remove(idApp+"txDutyCycle");
				m_valueRangeDictionary.remove(idApp+"txNumCyclesCephasonics");
				m_valueRangeDictionary.remove(idApp+"txNumCyclesManual");
				m_valueRangeDictionary.remove(idApp+"numScanlinesX");
				m_valueRangeDictionary.remove(idApp+"numScanlinesY");
				m_valueRangeDictionary.remove(idApp+"rxScanlineSubdivisionX");
				m_valueRangeDictionary.remove(idApp+"rxScanlineSubdivisionY");
				m_valueRangeDictionary.remove(idApp+"fovX");
				m_valueRangeDictionary.remove(idApp+"fovY");
				m_valueRangeDictionary.remove(idApp+"apertureSizeX");
				m_valueRangeDictionary.remove(idApp+"apertureSizeY");
				m_valueRangeDictionary.remove(idApp+"txApertureSizeX");
				m_valueRangeDictionary.remove(idApp+"txApertureSizeY");
				m_valueRangeDictionary.remove(idApp+"txFocusActive");
				m_valueRangeDictionary.remove(idApp+"txFocusDepth");
				m_valueRangeDictionary.remove(idApp+"txFocusWidth");
				m_valueRangeDictionary.remove(idApp+"txCorrectMatchingLayers");
				m_valueRangeDictionary.remove(idApp+"numSamplesRecon");
				m_valueRangeDictionary.remove(idApp+"steerNumAnglesX");
				m_valueRangeDictionary.remove(idApp+"steerAngleStartX");
				m_valueRangeDictionary.remove(idApp+"steerAngleEndX");
		}


		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			std::string idApp = getBeamSequenceApp(m_numBeamSequences,numSeq);

			// make nice string for GUI descriptors
			std::string descApp;
			if (m_numBeamSequences == 1)
			{
				descApp == "";
			}
			else
			{
				descApp = "Seq "+std::to_string(numSeq)+": ";
			}

			// overall scan type for sequence
			m_valueRangeDictionary.set<string>(idApp+"scanType", {"linear", "phased", "biphased", "planewave"}, "linear", descApp+"Scan Type");

			// beam ensemble specific settings
			m_valueRangeDictionary.set<double>(idApp+"txVoltage", 6, 60, 6, descApp+"Pulse voltage [V]");
			m_valueRangeDictionary.set<double>(idApp+"txFrequency", 0.0, 1000.0, 100, descApp+"Pulse frequency [Hz]");
			m_valueRangeDictionary.set<double>(idApp+"txPulseRepetitionFrequency", 0.0, 10000.0, 0.0, descApp+"Pulse repetition frequency [Hz]");
			m_valueRangeDictionary.set<double>(idApp+"txDutyCycle", 0.0, 1.0, 1.0, descApp+"Duty cycle [percent]");

			// further global parameters for one imaging sequence
			m_valueRangeDictionary.set<uint32_t>(idApp+"txNumCyclesCephasonics", 1, 10, 1, descApp+"Number Pulse Cycles (ceph)");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txNumCyclesManual", 1, 10, 1, descApp+"Number Pulse Cycles (manual)");

			// beam specific settings
			m_valueRangeDictionary.set<uint32_t>(idApp+"numScanlinesX", 1, 512, 32, descApp+"Number of scanlines X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"numScanlinesY", 1, 512, 1, descApp+"Number of scanlines Y");
			m_valueRangeDictionary.set<uint32_t>(idApp+"rxScanlineSubdivisionX", 1, 512, 32, descApp+"Rx scanline supersampling X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"rxScanlineSubdivisionY", 1, 512, 1, descApp+"Rx scanline supersampling Y");
			m_valueRangeDictionary.set<double>(idApp+"fovX", -178, 178, 60, descApp+"FOV X [degree]");
			m_valueRangeDictionary.set<double>(idApp+"fovY", -178, 178, 60, descApp+"FOV Y [degree]");
			m_valueRangeDictionary.set<uint32_t>(idApp+"apertureSizeX", 0, 384, 0, descApp+"Aperture X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"apertureSizeY", 0, 384, 0, descApp+"Aperture Y");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txApertureSizeX", 0, 384, 0, descApp+"TX Aperture X");
			m_valueRangeDictionary.set<uint32_t>(idApp+"txApertureSizeY", 0, 384, 0, descApp+"TX Aperture Y");
			m_valueRangeDictionary.set<string>(idApp+"txWindowType", {"Rectangular", "Hann", "Hamming","Gauss"}, "Rectangular", descApp+"TX apodization");
			m_valueRangeDictionary.set<double>(idApp+"txWindowParameter", 0.0, 10.0, 0.0, descApp+"TxWindow parameter");
			m_valueRangeDictionary.set<bool>(idApp+"txFocusActive", {false, true}, true, descApp+"TX focus");
			m_valueRangeDictionary.set<double>(idApp+"txFocusDepth", 0.0, 300.0, 50.0, descApp+"Focus depth [mm]");
			m_valueRangeDictionary.set<double>(idApp+"txFocusWidth", 0.0, 20.0, 0.0, descApp+"Focus width [mm]");
			m_valueRangeDictionary.set<bool>(idApp+"txCorrectMatchingLayers", {true, false}, false, descApp+"TX matching layer correction");
			m_valueRangeDictionary.set<uint32_t>(idApp+"numSamplesRecon", 10, 4096, 1000, descApp+"Number of samples Recon");
			m_valueRangeDictionary.set<uint32_t>(idApp+"steerNumAnglesX", 1, 256, 1, descApp+"Number of steered beam angles X");
			m_valueRangeDictionary.set<double>(idApp+"steerAngleStartX", -90.0, 90.0, 0.0, descApp+"Angle of first steered beam X [deg]");
			m_valueRangeDictionary.set<double>(idApp+"steerAngleEndX", -90.0, 90.0, 0.0, descApp+"Angle of last steered beam X [deg]");
		}
	}

	void UsIntDemonstrator::checkOptions()
	{
	}

	void UsIntDemonstrator::updateTransducer() {
		//create new transducer
		vec2s maxAperture{0,0};
		if (m_probeName == "LEDs") {
			double probePitch = 20;
			m_pTransducer = unique_ptr<USTransducer>(
					new USTransducer(
							USDEMONSTRATOR_NUM_PINS,
							vec2s{USDEMONSTRATOR_NUM_PINS,1},
							USTransducer::Linear,
							vector<double>(USDEMONSTRATOR_NUM_PINS - 1, probePitch),
							vector<double>(0),
							vector<std::pair<double, double> >{}));

			maxAperture = {USDEMONSTRATOR_NUM_PINS,1};


		} 
		else if (m_probeName == "Demonstrator")
		{
			// Linear array with 64 elements
			double probePitch = 15.24;
			m_pTransducer = unique_ptr<USTransducer>(
					new USTransducer(
							USDEMONSTRATOR_NUM_PINS,
							vec2s{USDEMONSTRATOR_NUM_PINS,1},
							USTransducer::Linear,
							vector<double>(USDEMONSTRATOR_NUM_PINS - 1, probePitch),
							vector<double>(0),
							vector<std::pair<double, double> >{})
						);

			maxAperture = {USDEMONSTRATOR_NUM_PINS,1};

		}
		m_numMuxedChannels = USDEMONSTRATOR_NUM_PINS;

		// we have a single "m_numMuxedChannels" channel system without muxing
		m_probeElementsToMuxedChannelIndices.resize(m_numMuxedChannels);

		for(size_t probeElem = 0; probeElem < m_numMuxedChannels; probeElem++)
		{
			m_probeElementsToMuxedChannelIndices[probeElem] = probeElem;
		}

		m_pSequencer->setTransducer(m_pTransducer.get());


		// TODO: currently all beamformers share same aperture
		for (auto numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			// TODO: support steering also in y
			size_t numAnglesSeq = m_pSequencer->getNumAngles(numSeq).x;

			for (size_t angleSeq = 0; angleSeq < numAnglesSeq; ++angleSeq)
			{
				std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq,angleSeq);

				vec2s bfTxApertureSize = bf->getTxApertureSize();
				vec2s bfApertureSize = bf->getApertureSize();

				if(bfTxApertureSize.x == 0 || bfTxApertureSize.y == 0)
				{
					bfTxApertureSize = maxAperture;
				}
				if(bfApertureSize.x == 0 || bfApertureSize.y == 0)
				{
					bfApertureSize = maxAperture;
				}
				bfApertureSize = min(bfApertureSize, maxAperture);
				bfTxApertureSize = min(bfTxApertureSize, maxAperture);

				bf->setTxMaxApertureSize(bfTxApertureSize);
				bf->setMaxApertureSize(bfApertureSize);


				// (re)compute the internal TX parameters for a beamformer if any parameter changed
				if (!bf->isReady())
				{
					bf->computeTxParameters();
				}
			}
		}
	}

	//void UsIntDemonstrator::updateImageProperties() {

	//	// iterate over all defined beam sequences, each beam-sequ defines one USImageProperties object
	//	for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
	//	{
	//

	//		std::shared_ptr<const Beamformer> bf = m_pSequencer->getBeamformer(numSeq);
	//		std::shared_ptr<const USImageProperties> imageProps = m_pSequencer->getUSImgProperties(numSeq);

	//		vec2s numScanlines = bf->getNumScanlines();
	//		size_t numDepths = bf->getNumDepths();
	//		vec2s rxScanlines = bf->getNumRxScanlines();
	//		vec2 steeringAngle = bf->getTxSteeringAngle();
	//		vec2 sectorAngle = bf->getTxSectorAngle();
	//		vec2s apertureSize = bf->getApertureSize();
	//		vec2s txApertureSize = bf->getTxApertureSize();


	//		auto newProps = make_shared<USImageProperties>(
	//			numScanlines,
	//			numDepths,
	//			USImageProperties::ImageType::BMode,
	//			USImageProperties::ImageState::Raw,
	//			USImageProperties::TransducerType::Linear,
	//			m_endDepth );

	//		newProps->setImageType(USImageProperties::ImageType::BMode);				// Defines the type of information contained in the image
	//		newProps->setImageState(USImageProperties::ImageState::Raw);				// Describes the state the image is currently in
	//		newProps->setScanlineLayout(numScanlines);					// number of scanlines acquired
	//		newProps->setDepth(m_endDepth);								// depth covered

	//		/* imageProps->setNumSamples(m_num);								// number of samples acquired on each scanline */
	//		/* imageProps->setImageResolution(double resolution);  			// the resolution of the scanConverted image */

	//		// Defines the type of transducer
	//		if (m_probeName == "Linear" || m_probeName == "CPLA12875" || m_probeName == "CPLA06475") {
	//			newProps->setTransducerType(USImageProperties::TransducerType::Linear);
	//		} else if (m_probeName == "2D") {
	//			newProps->setTransducerType(USImageProperties::TransducerType::Bicurved);
	//		}


	//		// publish Rx Scanline parameters together with the RawData
	//		if(imageProps && imageProps->getScanlineInfo())
	//		{
	//			newProps->setScanlineInfo(imageProps->getScanlineInfo());
	//		}

	//		// geometrical beamformer-related settings
	//		newProps->setSpecificParameter("UsIntCepCc.numScanlines.x", numScanlines.x);
	//		newProps->setSpecificParameter("UsIntCepCc.numScanlines.y", numScanlines.y);
	//		newProps->setSpecificParameter("UsIntCepCc.rxScanlines.x", rxScanlines.x);
	//		newProps->setSpecificParameter("UsIntCepCc.rxScanlines.y", rxScanlines.y);
	//		newProps->setSpecificParameter("UsIntCepCc.txSteeringAngle.x", steeringAngle.x);
	//		newProps->setSpecificParameter("UsIntCepCc.txSteeringAngle.y", steeringAngle.y);
	//		newProps->setSpecificParameter("UsIntCepCc.txSectorAngle.x", sectorAngle.x);
	//		newProps->setSpecificParameter("UsIntCepCc.txSectorAngle.y", sectorAngle.y);

	//		newProps->setSpecificParameter("UsIntCepCc.apertureSize.x", apertureSize.x);
	//		newProps->setSpecificParameter("UsIntCepCc.apertureSize.y", apertureSize.y);
	//		newProps->setSpecificParameter("UsIntCepCc.txApertureSize.x", txApertureSize.x);
	//		newProps->setSpecificParameter("UsIntCepCc.txApertureSize.y", txApertureSize.y);
	//		newProps->setSpecificParameter("UsIntCepCc.txFocusActive", bf->getTxFocusActive());
	//		newProps->setSpecificParameter("UsIntCepCc.txFocusDepth", bf->getTxFocusDepth());
	//		newProps->setSpecificParameter("UsIntCepCc.txFocusWidth", bf->getTxFocusWidth());
	//		newProps->setSpecificParameter("UsIntCepCc.txCorrectMatchingLayers", bf->getTxCorrectMatchingLayers());
	//		newProps->setSpecificParameter("UsIntCepCc.numSamplesRecon", bf->getNumDepths());
	//		newProps->setSpecificParameter("UsIntCepCc.scanType", bf->getScanType());


	//		// setting specific for beam ensemble transmit, not handled not within beamformer
	//		newProps->setSpecificParameter("UsIntCepCc.txFrequency",m_beamEnsembleTxParameters.at(numSeq).txFrequency);
	//		newProps->setSpecificParameter("UsIntCepCc.txPrf", m_beamEnsembleTxParameters.at(numSeq).txPrf);
	//		newProps->setSpecificParameter("UsIntCepCc.txVoltage", m_beamEnsembleTxParameters.at(numSeq).txVoltage);


	//		newProps->setSpecificParameter("UsIntCepCc.txNumCyclesCephasonics",  m_beamEnsembleTxParameters.at(numSeq).txNumCyclesCephasonics);
	//		newProps->setSpecificParameter("UsIntCepCc.txNumCyclesManual", m_beamEnsembleTxParameters.at(numSeq).txNumCyclesManual);


	//		//publish system-wide parameter settings to properties object
	//		newProps->setSpecificParameter("UsIntCepCc.systemTxClock", m_systemTxClock);
	//		newProps->setSpecificParameter("UsIntCepCc.probeName", m_probeName);
	//		newProps->setSpecificParameter("UsIntCepCc.startDepth", m_startDepth);
	//		newProps->setSpecificParameter("UsIntCepCc.endDepth", m_endDepth);
	//		newProps->setSpecificParameter("UsIntCepCc.processorMeasureThroughput", m_processorMeasureThroughput);
	//		newProps->setSpecificParameter("UsIntCepCc.speedOfSound", m_speedOfSound);

	//		newProps->setSpecificParameter("UsIntCepCc.tgc", m_vgaGain);
	//		newProps->setSpecificParameter("UsIntCepCc.decimation", m_decimation);
	//		newProps->setSpecificParameter("UsIntCepCc.decimationFilterBypass", m_decimationFilterBypass);
	//		newProps->setSpecificParameter("UsIntCepCc.antiAliasingFilterFrequency", m_antiAliasingFilterFrequency);
	//		newProps->setSpecificParameter("UsIntCepCc.highPassFilterBypass", m_highPassFilterBypass);
	//		newProps->setSpecificParameter("UsIntCepCc.highPassFilterFrequency", m_highPassFilterFrequency);
	//		newProps->setSpecificParameter("UsIntCepCc.lowNoiseAmplifierGain", m_lowNoiseAmplifierGain);
	//		newProps->setSpecificParameter("UsIntCepCc.inputImpedance", m_inputImpedance);

	//		m_pSequencer->setUSImgProperties(numSeq, newProps);
	//	}
	//}





	void UsIntDemonstrator::initializeDevice()
	{
		m_interface = std::unique_ptr<UsDemonstrator>(new UsDemonstrator());
		m_interface->setCallback([this](const std::vector<std::vector<uint8_t> >& im) { putData(im); });
		//Step 1 ------------------ "Setup Platform"
		//m_cPlatformHandle = setupPlatform();

		//checkOptions();

		//m_cUSEngine = unique_ptr<USEngine>(new USEngine(*m_cPlatformHandle));
		//m_cUSEngine->stop();
		//m_cUSEngine->setBlocking(false);

		//Step 2 ----------------- "Create Scan Definition"
		setupScan();

		//Step 3 ----------------- "Create Ultrasound Engine Thread"
		//create the data processor that later handles the data
		/*m_pDataProcessor = unique_ptr<UsIntDemonstratorProc>(
			new UsIntDemonstratorProc(*m_cPlatformHandle, this)
			);
		if(m_processorMeasureThroughput)
		{
			m_pDataProcessor->setMeasureThroughput(true, 50000);
		}
		//Create execution thread to run USEngine
		m_runEngineThread = thread([this]() {
			//The run function of USEngine starts its internal state machine that will run infinitely
			//until the USEngine::teardown() function is called or a fatal exception.
			m_cUSEngine->run(*m_pDataProcessor);

			//This thread will only return null on teardown of USEngine.
		});

		std::this_thread::sleep_for (std::chrono::seconds(2));*/
		logging::log_log("USEngine: initialized");

		m_ready = true;
	}

	std::vector<uint8_t> UsIntDemonstrator::createWeightedWaveform(
		const BeamEnsembleTxParameters& txParams, size_t numTotalEntries, float weight, size_t delaySamples, size_t maxDelaySamples, uint8_t csTxOversample)
	{
		//Creating TX single pulse
		//The wave pulse is constructed at 4x the system clock.
		//Assuming the system clock is 40MHz, then below is an even duty cycle
		//32 positive, 32 negative pulse waveform.  NOTE:  The leading, trailing, and middle ground
		//points are required in order to make a proper wave.
		//The Wave Frequency is given by TXFREQ = SysClock*4/Number of pos pulses + Number of neg pulses
		//2.5MHz Transmit Pulse Signal = 40*4/(32+32)
		//Calculation of num_pules to make frequency of transmit pulse is based on _txFreq, set by -f option
		size_t pulseHalfLength = static_cast<size_t>((
				static_cast<double>(m_systemTxClock)*
				csTxOversample /
				(txParams.txFrequency)
				)/2);
		size_t pulseHalfLengthWeighted = static_cast<size_t>(std::round(std::max((
				weight*static_cast<double>(m_systemTxClock)*
				csTxOversample /
				(txParams.txFrequency)
				)/2.0, 0.0)));

		vector<uint8_t> waveDef((pulseHalfLength*2) * txParams.txNumCyclesManual + 1 + maxDelaySamples, 0);
		for(size_t cycleIdx = 0; cycleIdx < txParams.txNumCyclesManual; cycleIdx++)
		{
			//Points to element with Leading Ground
			size_t firstIdx = cycleIdx*(pulseHalfLength*2) + delaySamples;
			//Points to element with Mid Ground
			size_t centerIdx = firstIdx + pulseHalfLength;

			for (size_t i = 1; i <= pulseHalfLengthWeighted; i++)
			{
				waveDef[centerIdx - i] = 1;
				waveDef[centerIdx + i] = 0;
			}
		}

		return waveDef;
	}

	void UsIntDemonstrator::startAcquisition()
	{
		m_interface->start();

		m_interfaceRunnerThread = std::thread([this] {m_interface->executionLoop(); });
	}

	void UsIntDemonstrator::stopAcquisition()
	{
		m_interface->stop();
	}

	void UsIntDemonstrator::configurationDictionaryChanged(const ConfigurationDictionary& newConfig)
	{
		// check if number of beam sequences has changed in the new configuration
		// if it did, update the value ranges for each sequence
		size_t numBeamSequences = newConfig.get<uint32_t>("sequenceNumFrames", 1);
		if (m_numBeamSequences != numBeamSequences)
		{
			logging::log_log("UsIntCephasonics: New number of beam sequences ", numBeamSequences, ", was formerly ", m_numBeamSequences);
			size_t oldNumBeamSequences = m_numBeamSequences;
			m_numBeamSequences = numBeamSequences;
			setBeamSequenceValueRange(oldNumBeamSequences);

			// create new sequencer
			m_pSequencer = unique_ptr<Sequencer>(new Sequencer(m_numBeamSequences));
			m_beamEnsembleTxParameters.resize(m_numBeamSequences);
		}
	}

	// change of multiple values in configuration result in a major (and re-initialization) update of all values
	void UsIntDemonstrator::configurationChanged()
	{
		// update systemwide configuration values
		m_systemTxClock = m_configurationDictionary.get<uint32_t>("systemTxClock");
		m_probeName = m_configurationDictionary.get<string>("probeName");
		m_endDepth = m_configurationDictionary.get<double>("endDepth");
		m_speedOfSound = m_configurationDictionary.get<double>("speedOfSound");

		// iterate over all defined beam sequences, each beam-sequ defines one USImageProperties object
		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			size_t numAnglesSeq = m_pSequencer->getNumAngles(numSeq).x;
			for (size_t angleSeq = 0; angleSeq < numAnglesSeq; ++angleSeq)
			{
				std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq, angleSeq);
				std::string seqIdApp = getBeamSequenceApp(m_numBeamSequences,numSeq);

				// scan or image-specific configuration values
				std::string scanType = m_configurationDictionary.get<std::string>(seqIdApp+"scanType");
				bf->setScanType(scanType);

				bf->setTxFocusActive(m_configurationDictionary.get<bool>(seqIdApp+"txFocusActive"));
				bf->setTxFocusDepth(m_configurationDictionary.get<double>(seqIdApp+"txFocusDepth"));
				bf->setRxFocusDepth(m_configurationDictionary.get<double>(seqIdApp+"txFocusDepth")); // currently rx and tx focus are the same
				bf->setTxFocusWidth(m_configurationDictionary.get<double>(seqIdApp+"txFocusWidth"));
				bf->setTxCorrectMatchingLayers(m_configurationDictionary.get<bool>(seqIdApp+"txCorrectMatchingLayers"));
				bf->setNumDepths(m_configurationDictionary.get<uint32_t>(seqIdApp+"numSamplesRecon"));

				bf->setSpeedOfSound(m_speedOfSound);
				bf->setDepth(m_endDepth);

				vec2s numScanlines;
				numScanlines.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"numScanlinesX");
				numScanlines.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"numScanlinesY");
				bf->setNumScanlines(numScanlines);

				vec2s rxScanlinesSubdivision;
				rxScanlinesSubdivision.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionX");
				rxScanlinesSubdivision.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionY");
				bf->setRxScanlineSubdivision( rxScanlinesSubdivision );

				vec2 fov;
				fov.x = m_configurationDictionary.get<double>(seqIdApp+"fovX");
				fov.y = m_configurationDictionary.get<double>(seqIdApp+"fovY");
				bf->setFov(fov);

				vec2s apertureSize;
				apertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeX");
				apertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeY");
				bf->setMaxApertureSize(apertureSize);

				vec2s txApertureSize;
				txApertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeX");
				txApertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeY");
				bf->setTxMaxApertureSize(txApertureSize);

				string windowType = m_configurationDictionary.get<string>(seqIdApp+"txWindowType");
				bf->setTxWindowType(windowType);

				double winParam = m_configurationDictionary.get<double>("txWindowParameter");
				bf->setWindowParameter(winParam);



				// Todo: support steering also in y
				m_pSequencer->setNumAngles(numSeq, {m_configurationDictionary.get<uint32_t>(seqIdApp+"steerNumAnglesX"), 1});
				m_pSequencer->setStartAngle(numSeq, { m_configurationDictionary.get<double>(seqIdApp+"steerAngleStartX"), 0.0 });
				m_pSequencer->setEndAngle(numSeq, { m_configurationDictionary.get<double>(seqIdApp+"steerAngleEndX"), 0.0 });


				// ensemble-specific parameters (valid for a whole image irrespective of whether it is linear, phased, planewave, or push)
				m_beamEnsembleTxParameters.at(numSeq).txPrf = m_configurationDictionary.get<double>(seqIdApp+"txPulseRepetitionFrequency");
				m_beamEnsembleTxParameters.at(numSeq).txVoltage = m_configurationDictionary.get<double>(seqIdApp+"txVoltage");
				m_beamEnsembleTxParameters.at(numSeq).txFrequency = m_configurationDictionary.get<double>(seqIdApp+"txFrequency");
				m_beamEnsembleTxParameters.at(numSeq).txDutyCycle = m_configurationDictionary.get<double>(seqIdApp+"txDutyCycle");
				m_beamEnsembleTxParameters.at(numSeq).txNumCyclesManual = m_configurationDictionary.get<uint32_t>(seqIdApp+"txNumCyclesManual");
			}
		}

		updateTransducer();

		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			size_t numAnglesSeq = m_pSequencer->getNumAngles(numSeq).x;
			for (size_t angleSeq = 0; angleSeq < numAnglesSeq; ++angleSeq)
			{
				std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq, angleSeq);

				// (re)compute the internal TX parameters for a beamformer if any parameter changed
				if (!bf->isReady())
				{
					bf->computeTxParameters();
				}
			}
		}

		//updateImageProperties();
	}

	void UsIntDemonstrator::configurationEntryChanged(const std::string & configKey)
	{
		lock_guard<mutex> lock(m_objectMutex);

		// settings which can be changed online
		if(m_ready)
		{

		}

		//these properties require large reconfigurations. for now allow them only before initialization
		if(!m_ready)
		{
			// global (system-wide) settings

			if(configKey == "endDepth")
			{
				m_endDepth = m_configurationDictionary.get<double>("endDepth");
			}
			if(configKey == "speedOfSound")
			{
				m_speedOfSound = m_configurationDictionary.get<double>("speedOfSound");
			}

			// local settings (per firing)
			for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
			{
				size_t numAnglesSeq = m_pSequencer->getNumAngles(numSeq).x;
				std::string seqIdApp = getBeamSequenceApp(m_numBeamSequences,numSeq);

				for (size_t angleSeq = 0; angleSeq < numAnglesSeq; ++angleSeq)
				{
					std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq, angleSeq);

					if(configKey == seqIdApp+"numScanlinesX" || configKey == seqIdApp+"numScanlinesY")
					{
						vec2s numScanlines;
						numScanlines.x = m_configurationDictionary.get<size_t>(seqIdApp+"numScanlinesX");
						numScanlines.y = m_configurationDictionary.get<size_t>(seqIdApp+"numScanlinesY");
						bf->setNumScanlines( numScanlines );
					}
					if(configKey == seqIdApp+"rxScanlineSubdivisionX" || configKey == seqIdApp+"rxScanlineSubdivisionY")
					{
						vec2s rxScanlinesSubdivision;
						rxScanlinesSubdivision.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionX");
						rxScanlinesSubdivision.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"rxScanlineSubdivisionY");
						bf->setRxScanlineSubdivision( rxScanlinesSubdivision );
					}

					if(configKey == seqIdApp+"fovX" || configKey == seqIdApp+"fovY")
					{
						vec2 fov;
						fov.x = m_configurationDictionary.get<double>(seqIdApp+"fovX");
						fov.y = m_configurationDictionary.get<double>(seqIdApp+"fovY");
						bf->setFov( fov );
					}
					if(configKey == seqIdApp+"apertureSizeX" || configKey == seqIdApp+"apertureSizeY")
					{
						vec2s apertureSize;
						apertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeX");
						apertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"apertureSizeY");
						bf->setMaxApertureSize( apertureSize );
					}
					if(configKey == seqIdApp+"txApertureSizeX" || configKey == seqIdApp+"txApertureSizeY")
					{
						vec2s txApertureSize;
						txApertureSize.x = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeX");
						txApertureSize.y = m_configurationDictionary.get<uint32_t>(seqIdApp+"txApertureSizeY");
						bf->setTxMaxApertureSize( txApertureSize );
					}
					if(configKey == seqIdApp+"txFocusActive")
					{
						bf->setTxFocusActive(m_configurationDictionary.get<bool>(seqIdApp+"txFocusActive"));
					}
					if(configKey == seqIdApp+"txFocusDepth")
					{
						bf->setTxFocusDepth(m_configurationDictionary.get<double>(seqIdApp+"txFocusDepth"));
					}
					if(configKey == seqIdApp+"txFocusWidth")
					{
						bf->setTxFocusWidth(m_configurationDictionary.get<double>(seqIdApp+"txFocusWidth"));
					}
					if(configKey == seqIdApp+"txCorrectMatchingLayers")
					{
						bf->setTxCorrectMatchingLayers(m_configurationDictionary.get<bool>(seqIdApp+"txCorrectMatchingLayers"));
					}
					if(configKey == seqIdApp+"numSamplesRecon")
					{
						bf->setNumDepths(m_configurationDictionary.get<uint32_t>(seqIdApp+"numSamplesRecon"));
					}
					if(configKey == seqIdApp+"steerNumAnglesX")
					{
						m_pSequencer->setNumAngles(numSeq, {m_configurationDictionary.get<uint32_t>(seqIdApp+"steerNumAnglesX"), 1});
					}
					if(configKey == seqIdApp+"steerAngleStartX")
					{
						m_pSequencer->setStartAngle(numSeq, { m_configurationDictionary.get<double>(seqIdApp+"steerAngleStartX"),0 });
					}
					if(configKey == seqIdApp+"steerAngleEndX")
					{
						m_pSequencer->setEndAngle(numSeq, {m_configurationDictionary.get<double>(seqIdApp+"steerAngleEndX"),0 });
					}

					if(configKey == seqIdApp+"txDutyCycle")
					{
						// TODO
					}
					if(configKey == seqIdApp+"txFrequency")
					{
						m_beamEnsembleTxParameters.at(numSeq).txFrequency = m_configurationDictionary.get<double>(seqIdApp+"txFrequency");
					}
					if(configKey == seqIdApp+"txPulseRepetitionFrequency")
					{
						m_beamEnsembleTxParameters.at(numSeq).txPrf = m_configurationDictionary.get<double>(seqIdApp+"txPulseRepetitionFrequency");
					}
					if(configKey == seqIdApp+"txNumCyclesManual")
					{
						m_beamEnsembleTxParameters.at(numSeq).txNumCyclesManual = m_configurationDictionary.get<uint32_t>(seqIdApp+"txNumCyclesManual");
					}

					// update bf internal parameters if a relevant parameter has changed
					if (!bf->isReady())
					{
						bf->computeTxParameters();
					}
				}
			}

		}
		//updateImageProperties();
	}

	// A scan consist of the overall scanning sequence and can consit of multiple frames to be acquire sequentially/periodically
	void UsIntDemonstrator::setupScan()
	{
		//create new transducer
		updateTransducer();

		// sets up the transducer, beamformer and creates the FrameDef
		createSequence();
	}


	void UsIntDemonstrator::createSequence()
	{

		for (size_t numSeq = 0; numSeq < m_numBeamSequences; ++numSeq)
		{
			// TODO: support steering also in y
			size_t numAnglesSeq = m_pSequencer->getNumAngles(numSeq).x;

			for (size_t angleSeq = 0; angleSeq < numAnglesSeq; ++angleSeq)
			{
				std::shared_ptr<Beamformer> bf = m_pSequencer->getBeamformer(numSeq,angleSeq);
				std::shared_ptr<USImageProperties> props = m_pSequencer->getUSImgProperties(numSeq,angleSeq);

				// push rx parameters to US properties
				props->setScanlineInfo(bf->getRxParameters());

				// Get the Tx scanline parameters to program the Hardware with them
				const std::vector<ScanlineTxParameters3D>* beamTxParams = bf->getTxParameters();

				std::pair<size_t, UsDemonstrator::Frame> fdef = createFrame(beamTxParams, props, m_beamEnsembleTxParameters.at(numSeq));

				// store framedef and add it to Cephasonics interface
				m_pFrameMap[fdef.first] = std::pair<size_t,size_t>(numSeq,angleSeq);
				m_pFrameDefs.push_back(fdef.second);

				m_interface->addFrame(fdef.second);
			}
		}
	}

	// a frame is defined as set of BeamEnsembles. Multiple frames can be put together into a sequence
	// SUPRA Frames have a 1:1 correspondene to a beamformer, i.e. a Beamformer is generating necessary info for a frame
	std::pair<size_t, UsDemonstrator::Frame> UsIntDemonstrator::createFrame(
		const std::vector<ScanlineTxParameters3D>* txBeamParams, 
		const std::shared_ptr<USImageProperties> imageProps, 
		const BeamEnsembleTxParameters& txEnsembleParams)
	{
		static size_t framesCreated = 0;

		// publish Rx Scanline parameters together with the RawData
		// updateImageProperties();


		// scanlines
		vec2s numScanlines = imageProps->getScanlineLayout();

		UsDemonstrator::Frame frame;
		for(auto txParams: *txBeamParams)
		{
			// create the beam ensemble for this txBeam
			UsDemonstrator::TransmitBeam ensembleDef = createTransmitBeamFromScanlineTxParameter(txEnsembleParams, numScanlines, txParams);
			frame.push_back(ensembleDef);
		}

		size_t newFrameID = framesCreated;
		framesCreated++;

		return std::make_pair(newFrameID,frame);
	}

	UsDemonstrator::TransmitBeam UsIntDemonstrator::createTransmitBeamFromScanlineTxParameter(
		const BeamEnsembleTxParameters& txEnsembleParameters, 
		const vec2s numScanlines, 
		const ScanlineTxParameters3D& txParameters)
	{
		//Creating TX single pulse
		//The wave pulse is constructed at 4x the system clock.
		//Assuming the system clock is 40MHz, then below is an even duty cycle
		//32 positive, 32 negative pulse waveform.  NOTE:  The leading, trailing, and middle ground
		//points are required in order to make a proper wave.
		//The Wave Frequency is given by TXFREQ = SysClock*4/Number of pos pulses + Number of neg pulses
		//2.5MHz Transmit Pulse Signal = 40*4/(32+32)
		//Calculation of num_pules to make frequency of transmit pulse is based on _txFreq, set by -f option
		size_t pulseHalfLength = static_cast<size_t>((
				static_cast<double>(m_systemTxClock) /
				(txEnsembleParameters.txFrequency)
				)/2);

		// find maximum delay
		double maxDelay = 0;
		for (auto& delayVect : txParameters.delays)
		{
			for (auto& delay : delayVect)
			{
				maxDelay = max(maxDelay, delay);
			}
		}
		size_t maxDelaySamples = static_cast<size_t>(round(maxDelay * m_systemTxClock));

		size_t numTotalEntries = pulseHalfLength*2 * txEnsembleParameters.txNumCyclesManual + 1 + maxDelaySamples;

		// create passive wave with equal length
		vector<uint8_t> myWaveDef_passive(numTotalEntries, 0);

		vec2s elementLayout = m_pTransducer->getElementLayout();
		// create tx map w.r.t. the muxed channels
		vector<bool> txMap(m_numMuxedChannels, false);
		// the transmit delay vector also has to be specified for all muxed channels
		vector<double> txDelays(m_numMuxedChannels, 0);
		vector<vector<uint8_t> > txWaves(m_numMuxedChannels, myWaveDef_passive);
		for (size_t activeElementIdxX = txParameters.firstActiveElementIndex.x; activeElementIdxX <= txParameters.lastActiveElementIndex.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = txParameters.firstActiveElementIndex.y; activeElementIdxY <= txParameters.lastActiveElementIndex.y; activeElementIdxY++)
			{
				if(txParameters.elementMap[activeElementIdxX][activeElementIdxY]) //should be true all the time, except we explicitly exclude elements
				{
					size_t muxedChanIdx = m_probeElementsToMuxedChannelIndices[activeElementIdxX + elementLayout.x*activeElementIdxY];
					txMap[muxedChanIdx] = true;
				}
			}
		}

		for (size_t activeElementIdxX = txParameters.txAperture.begin.x; activeElementIdxX <= txParameters.txAperture.end.x; activeElementIdxX++)
		{
			for (size_t activeElementIdxY = txParameters.txAperture.begin.y; activeElementIdxY <= txParameters.txAperture.end.y; activeElementIdxY++)
			{
				if(txParameters.txElementMap[activeElementIdxX][activeElementIdxY])
				{
					size_t localElementIdxX = activeElementIdxX -txParameters.txAperture.begin.x;
					size_t localElementIdxY = activeElementIdxY -txParameters.txAperture.begin.y;
					size_t muxedChanIdx = m_probeElementsToMuxedChannelIndices[activeElementIdxX + elementLayout.x*activeElementIdxY];

					//TX Delays are given in units of TX_CLOCK
					// -> convert from seconds to TX_Clock
					size_t txDelaySamples  = static_cast<size_t>(round(
						txParameters.delays[localElementIdxX][localElementIdxY]*static_cast<double>(m_systemTxClock))); // from seconds to samples

					float txWeight = static_cast<float>(txParameters.weights[localElementIdxX][localElementIdxY]);
					txWaves[muxedChanIdx] = createWeightedWaveform(txEnsembleParameters, numTotalEntries, txWeight, txDelaySamples, maxDelaySamples, 1);
				}
				else {
					//			txWaves[elemIdx] = myWaveDef_passive;
				}
			}
		}

		UsDemonstrator::TransmitBeam txBeam;
		txBeam.txWaves = txWaves;
		txBeam.txMap = txMap;

		//const BeamDef* txBeam = &BeamDef::createTXBeamDef(
		//		*m_cPlatformHandle,
		//		txWaves,
		//		txDelays);
		//return txBeam;
		return txBeam;
	}

	bool UsIntDemonstrator::ready()
	{
		return m_ready;
	}

	void UsIntDemonstrator::freeze()
	{
		m_interface->freeze();
	}

	void UsIntDemonstrator::unfreeze()
	{
		m_interface->unfreeze();
	}


	void UsIntDemonstrator::putData(const std::vector<std::vector<uint8_t> >& image)
	{
		double timestamp = getCurrentTime();
		lock_guard<mutex> lock(m_objectMutex);
		m_callFrequency.measure();

		auto pData = make_shared<Container<uint8_t> >(LocationHost, ContainerFactory::getNextStream(), image.size()* image[0].size());

		size_t numSamples = image[0].size();
		size_t numVectors = image.size();
		size_t index = 0;
		for (size_t sample = 0; sample < numSamples; sample++)
		{
			for (size_t scanline = 0; scanline < numVectors; scanline++)
			{
				*(pData->get() + scanline + sample*numVectors) = image[scanline][sample] * 255;
			}
		}

		std::shared_ptr<USImageProperties> imProps = make_shared<USImageProperties>(vec2s{ numVectors, 1 }, numSamples,
			USImageProperties::ImageType::BMode, USImageProperties::ImageState::PreScan, 
			USImageProperties::TransducerType::Linear, static_cast<double>(numVectors));

		// we received the data from all necessary platforms, now we can start the beamforming
		shared_ptr<USImage<uint8_t> > rawData = make_shared<USImage<uint8_t> >
			(vec2s{ numVectors, numSamples },
				pData,
				imProps,
				timestamp,
				timestamp);

		addData<0>(rawData);
	}
}
