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


#ifndef __USINTDEMONSTRATOR_H__
#define __USINTDEMONSTRATOR_H__

#ifdef HAVE_DEVICE_DEMONSTRATOR

#include <atomic>
#include <memory>
#include <mutex>
#include <array>

#include <AbstractInput.h>
#include <USImage.h>
#include <fstream>

// should go to demonstrator class
namespace supra
{
	class UsDemonstrator
	{
	public:
		UsDemonstrator();

		struct TransmitBeam
		{
			std::vector<std::vector<uint8_t> > txWaves;
			std::vector<bool> txMap;
		};

		typedef std::vector<TransmitBeam> Frame;

		void addFrame(const Frame& frame);
		void executionLoop();
		void start();
		void stop();
		void freeze();
		void unfreeze();
		void setCallback(std::function<void(const std::vector<std::vector<uint8_t>> &)> callback);

	private:
		size_t m_txClock;
		double m_pulseFrequency;

		std::vector<Frame> m_frames;

		std::atomic<bool> m_running;
		std::atomic<bool> m_frozen;
		size_t m_currentFrame;
		size_t m_currentTransmit;

		std::function<void(const std::vector<std::vector<uint8_t>>&)> m_callback;

		std::mutex m_objMutex;
		SingleThreadTimer m_timer;
	};
}


namespace supra
{
	class Beamformer;
	class Sequencer;
	struct ScanlineTxParameters3D;
	class USTransducer;

	struct BeamEnsembleTxParameters
	{
		double txVoltage;				// voltage applied for pulse
		double txDutyCycle;				// duty cycle (percent) used for pulse
		double txFrequency;				// pulse frequency in MHz
		double txPrf;					// pulse repetition frequency of image in Hz

		size_t txNumCyclesManual;		// Manual pulse repetition during signal construction
	};


	class UsIntDemonstrator : public AbstractInput<RecordObject>
	{
	public:
		UsIntDemonstrator(tbb::flow::graph& graph, const std::string& nodeID);
		virtual ~UsIntDemonstrator();

		//Functions to be overwritten
	public:
		virtual void initializeDevice();
		virtual bool ready();

		virtual std::vector<size_t> getImageOutputPorts() { return{}; };
		virtual std::vector<size_t> getTrackingOutputPorts() { return{}; };

		virtual void freeze();
		virtual void unfreeze();

	protected:
		virtual void startAcquisition();
		//Needs to be thread safe
		virtual void stopAcquisition();
		//Needs to be thread safe
		virtual void configurationEntryChanged(const std::string& configKey);
		//Needs to be thread safe
		virtual void configurationChanged();
		// needs to be thread safe
		virtual void configurationDictionaryChanged(const ConfigurationDictionary& newConfig);

	private:
		void putData(const std::vector<std::vector<uint8_t> >& image);

		void checkOptions();
		void setupScan();
		//void initBeams();
		void createSequence();
		std::pair<size_t, UsDemonstrator::Frame> createFrame(const std::vector<ScanlineTxParameters3D>* txBeamParams, const std::shared_ptr<USImageProperties> imageProps, const BeamEnsembleTxParameters& txParamsCs);
		UsDemonstrator::TransmitBeam createTransmitBeamFromScanlineTxParameter(const BeamEnsembleTxParameters& txEnsembleParams, const vec2s numScanlines, const ScanlineTxParameters3D& txParameters);
		std::vector<uint8_t> createWeightedWaveform(const BeamEnsembleTxParameters& txParams, size_t numTotalEntries, float weight, size_t delaySamples, size_t maxDelaySamples, uint8_t csTxOversample);
		void updateTransducer();
		void setBeamSequenceValueRange(size_t oldBeamSequenceValueRange);
		std::string getBeamSequenceApp(size_t totalSequences, size_t sequenceId); 	// return string with appendix for each beam sequence configuration value

		std::mutex m_objectMutex;

		bool m_ready;

		std::unique_ptr<Sequencer> m_pSequencer;

		std::unique_ptr<USTransducer> m_pTransducer;

		// many Frame/SubFrames possible
		std::map<size_t, std::pair<size_t,size_t>> m_pFrameMap; // mapping of linearized frameIDs in cusdk to seq and angle number
		std::vector<UsDemonstrator::Frame> m_pFrameDefs;
		std::vector<BeamEnsembleTxParameters> m_beamEnsembleTxParameters; // CS specific transmit parameters

		// general system properties
		uint32_t m_numMuxedChannels;
		uint16_t m_probeMapping;

		// system-wide imaging settings (i.e. identical for all individual firings or images in a sequence)
		std::string m_probeName;
		std::vector<size_t> m_probeElementsToMuxedChannelIndices;
		double m_endDepth;
		double m_speedOfSound; // [m/s]

		// Todo Support generic number of imaging sequences in config directory and XML reader
		size_t m_numBeamSequences;

		// TX settings (system-wide)
		uint32_t m_systemTxClock;

		std::unique_ptr<UsDemonstrator> m_interface;
		std::thread m_interfaceRunnerThread;
	};
}

#endif //!HAVE_DEVICE_DEMONSTRATOR

#endif //!__USINTDEMONSTRATOR_H__
