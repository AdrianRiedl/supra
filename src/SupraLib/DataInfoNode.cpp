// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2018, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include "DataInfoNode.h"

#include "USImage.h"
#include "Beamformer/USRawData.h"
#include "SyncRecordObject.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
	DataInfoNode::DataInfoNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
	{
		// Create the underlying tbb node for handling the message passing. This usually does not need to be modified.
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeOneSidedQueueing>(
				new NodeTypeOneSidedQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj){
					// lock the object mutex to make sure no parameters are changed during processing
					unique_lock<mutex> l(m_mutex);
					return checkTypeAndProcess(inObj, 0);
			}));
		}
		else
		{
			m_node = unique_ptr<NodeTypeOneSidedDiscarding>(
				new NodeTypeOneSidedDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj){
					// lock the object mutex to make sure no parameters are changed during processing
					unique_lock<mutex> l(m_mutex);
					return checkTypeAndProcess(inObj, 0);
			}));
		}

		m_callFrequency.setName("DataInfoNode");

		// Define the parameters that this node reveals to the user
		m_valueRangeDictionary.set<string>("prefix", "Data", "Prefix");
		
		// read the configuration to apply the default values
		configurationChanged();
	}

	void DataInfoNode::configurationChanged()
	{
		m_prefix = m_configurationDictionary.get<string>("prefix");
	}

	void DataInfoNode::configurationEntryChanged(const std::string& configKey)
	{
		// lock the object mutex to make sure no processing happens during parameter changes
		unique_lock<mutex> l(m_mutex);
		if (configKey == "prefix")
		{
			m_prefix = m_configurationDictionary.get<string>("prefix");
		}
	}

	template <>
	void DataInfoNode::process<USImage>(std::shared_ptr<const RecordObject> object, size_t printOffset)
	{
		shared_ptr<const USImage> pInImage = dynamic_pointer_cast<const USImage>(object);
		if (pInImage)
		{
			auto size = pInImage->getSize();
			logging::log_info(m_prefix, std::string(printOffset, ' '), ": USImage, ", pInImage->getDataType(), ", ",
					"(", size.x, ", ", size.y, ", ", size.z, ")");
		}
		else {
			logging::log_error("DataInfoNode: '", m_prefix, "' could not cast object to USImage type");
		}
	}

	template <>
	void DataInfoNode::process<USRawData>(std::shared_ptr<const RecordObject> object, size_t printOffset)
	{
		shared_ptr<const USRawData> pInRawData = dynamic_pointer_cast<const USRawData>(object);
		if (pInRawData)
		{
			logging::log_info(m_prefix, std::string(printOffset, ' '), ": USRawData, ", pInRawData->getDataType(), ", ",
					"#scanlines: ", pInRawData->getNumScanlines(),
					"#elements: ", pInRawData->getNumElements(),
					"#channels: ", pInRawData->getNumReceivedChannels());
		}
		else {
			logging::log_error("DataInfoNode: '", m_prefix, "' could not cast object to USRawData type");
		}
	}

	template <>
	void DataInfoNode::process<SyncRecordObject>(std::shared_ptr<const RecordObject> object, size_t printOffset)
	{
		shared_ptr<const SyncRecordObject> pSyncObj = dynamic_pointer_cast<const SyncRecordObject>(object);
		if (pSyncObj)
		{
			logging::log_info(m_prefix, std::string(printOffset, ' '), ": SyncRecordObject with ", pSyncObj->getSyncedRecords().size(), " synced Objs");
			logging::log_info(m_prefix, std::string(printOffset, ' '), ": Main:");
			checkTypeAndProcess(pSyncObj->getMainRecord(), printOffset + 2);
			size_t children = 0;
			for (const auto& child : pSyncObj->getSyncedRecords())
			{
				logging::log_info(m_prefix, std::string(printOffset, ' '), ": child ", children++, ":");
				checkTypeAndProcess(child, printOffset + 2);
			}

		}
		else {
			logging::log_error("DataInfoNode: '", m_prefix, "' could not cast object to SyncRecordObject type");
		}
	}

	void DataInfoNode::checkTypeAndProcess(shared_ptr<const RecordObject> obj, size_t printOffset)
	{
		shared_ptr<USImage> pImage = nullptr;
		if (obj)
		{
			switch(obj->getType())
			{
				case TypeUSImage:
				{
					process<USImage>(obj, printOffset);
					break;
				}
				case TypeUSRawData:
				{
					process<USRawData>(obj, printOffset);
					break;
				}
				case TypeSyncRecordObject:
				{
					process<SyncRecordObject>(obj, printOffset);
					break;
				}
				case TypeTrackerDataSet:
				{
					logging::log_info(m_prefix, std::string(printOffset, ' '), ": TrackerDataSet");
					break;
				}
				case TypeRecordUnknown:
				default:
				{
					logging::log_info(m_prefix, std::string(printOffset, ' '), ": TypeRecordUnknown");
					break;
				}
			}
		}
	}
}
