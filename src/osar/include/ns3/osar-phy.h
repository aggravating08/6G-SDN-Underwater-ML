#ifndef OSAR_PHY_H
#define OSAR_PHY_H

#include "ns3/object.h"
#include "ns3/node.h"
#include <vector>

namespace ns3 {

class OsarPhy : public Object
{
public:
  static TypeId GetTypeId();
  OsarPhy();
  void SetNode(Ptr<Node> node);
  void PerformSpectrumSensing();
  std::vector<bool> GetIdleSubcarriers();

private:
  Ptr<Node> m_node;
  std::vector<bool> m_idleSubcarriers;
};

} // namespace ns3

#endif // OSAR_PHY_H

