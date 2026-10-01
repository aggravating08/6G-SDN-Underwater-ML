#ifndef OSAR_ROUTING_H
#define OSAR_ROUTING_H

#include "ns3/object.h"
#include "ns3/node.h"
#include "ns3/vector.h"

namespace ns3 {

class OsarRouting : public Object
{
public:
  static TypeId GetTypeId();
  OsarRouting();
  void SetNode(Ptr<Node> node);
  void SelectRelay(std::vector<Ptr<Node>> neighbors);

private:
  Ptr<Node> m_node;
};

} // namespace ns3

#endif // OSAR_ROUTING_H

