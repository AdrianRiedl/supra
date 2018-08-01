// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __STREAMSPLITNODE_H__
#define __STREAMSPLITNODE_H__

#include "AbstractNode.h"
#include "RecordObject.h"

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

namespace supra
{
	class StreamSplitNode : public AbstractNode {
	private:
		typedef tbb::flow::broadcast_node<std::shared_ptr<RecordObject> > SyncedOutputNodeType;
	public:
		StreamSplitNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 2; }

		virtual tbb::flow::graph_node * getInput(size_t index) {
			if (index == 0)
			{
				return m_node.get();
			}
			return nullptr;
		};

		virtual tbb::flow::graph_node * getOutput(size_t index) {
			if (index == 0)
			{
				return m_node.get();
			}
			if (index == 1)
			{
				return m_syncedOutputNode.get();
			}
			return nullptr;
		};
	protected:
		void configurationEntryChanged(const std::string& configKey);
		void configurationChanged();

	private:
		std::shared_ptr<RecordObject> checkTypeAndSplit(std::shared_ptr<RecordObject> mainObj);

		std::unique_ptr<tbb::flow::graph_node> m_node;
		std::unique_ptr<SyncedOutputNodeType> m_syncedOutputNode;
	};
}

#endif //!__STREAMSPLITNODE_H__
