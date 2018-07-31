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

#ifndef __DATAINFONODE_H__
#define __DATAINFONODE_H__

#include <memory>
#include <mutex>
#include <tbb/flow_graph.h>

#include <AbstractNode.h>
#include <RecordObject.h>
#include <Container.h>
#include <vec.h>

// To include the node fully, add it in src/SupraLib/CMakeLists.txt and "InterfaceFactory::createNode"!

namespace supra
{
	class DataInfoNode : public AbstractNode {

	public:
		DataInfoNode(tbb::flow::graph& graph, const std::string & nodeID, bool queueing);

		virtual size_t getNumInputs() { return 1; }
		virtual size_t getNumOutputs() { return 0; }

		virtual tbb::flow::graph_node * getInput(size_t index) {
			if (index == 0)
			{
				return m_node.get();
			}
			return nullptr;
		};

		virtual tbb::flow::graph_node * getOutput(size_t index) {
			return nullptr;
		};
	protected:
		void configurationEntryChanged(const std::string& configKey);
		void configurationChanged();

	private:
		template <typename T>
		void process(std::shared_ptr<const RecordObject> object, size_t printOffset);
		void checkTypeAndProcess(std::shared_ptr<const RecordObject> obj, size_t printOffset);

		std::unique_ptr<tbb::flow::graph_node> m_node;

		std::mutex m_mutex;

		std::string m_prefix;
	};
}

#endif //!__DATAINFONODE_H__
