#ifndef OSAR_MAC_H
#define OSAR_MAC_H

#include "ns3/object.h"
#include "ns3/node.h"
#include <map>

namespace ns3 {

class OsarMac : public Object
{
public:
  static TypeId GetTypeId();
  OsarMac();
  void SetNode(Ptr<Node> node);
  void BroadcastBeacon();
  void ReceiveBeacon(uint32_t senderId, std::vector<bool> channelState);

  std::map<uint32_t, std::vector<bool>> GetNeighborChannelStates();

private:
  Ptr<Node> m_node;
  std::map<uint32_t, std::vector<bool>> m_neighborStates;
};

} // namespace ns3

#endif // OSAR_MAC_H

