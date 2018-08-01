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

#include "StreamSplitNode.h"

#include "SyncRecordObject.h"
#include "Beamformer/USRawData.h"
#include <utilities/Logging.h>

using namespace std;

namespace supra
{
StreamSplitNode::StreamSplitNode(tbb::flow::graph & graph, const std::string & nodeID, bool queueing)
		: AbstractNode(nodeID, queueing)
	{
		if (queueing)
		{
			m_node = unique_ptr<NodeTypeQueueing>(
				new NodeTypeQueueing(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndSplit(inObj); }));
		}
		else
		{
			m_node = unique_ptr<NodeTypeDiscarding>(
				new NodeTypeDiscarding(graph, 1, [this](shared_ptr<RecordObject> inObj) -> shared_ptr<RecordObject> { return checkTypeAndSplit(inObj); }));
		}
		m_syncedOutputNode = unique_ptr<SyncedOutputNodeType>(new SyncedOutputNodeType(graph));
		m_callFrequency.setName(nodeID);
		configurationChanged();
	}

	void StreamSplitNode::configurationChanged()
	{
	}

	void StreamSplitNode::configurationEntryChanged(const std::string& configKey)
	{
	}
	shared_ptr<RecordObject> StreamSplitNode::checkTypeAndSplit(shared_ptr<RecordObject> inObj)
	{
		m_callFrequency.measure();

		auto returnObj = inObj;
		if (inObj && inObj->getType() == TypeSyncRecordObject)
		{
			shared_ptr<SyncRecordObject> inSynced = dynamic_pointer_cast<SyncRecordObject>(inObj);
			if (inSynced)
			{
				returnObj = inSynced->getMainRecord();

				if(inSynced->getSyncedRecords().size() >= 1)
				{
					m_syncedOutputNode->try_put(inSynced->getSyncedRecords()[0]);
				}
			}
		}
		return returnObj;
	}
}
