#include "ns3/osar-routing.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-application.h"
#include "ns3/osar-helper.h"
#include "ns3/osar-channel-model.h"
#include "ns3/osar-loss-model.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("OsarMac");

TypeId OsarMac::GetTypeId()
{
  static TypeId tid = TypeId("ns3::OsarMac")
    .SetParent<Object>()
    .SetGroupName("Osar")
    .AddConstructor<OsarMac>();
  return tid;
}

OsarMac::OsarMac() {}

void OsarMac::SetNode(Ptr<Node> node)
{
  m_node = node;
}

void OsarMac::BroadcastBeacon()
{
  NS_LOG_INFO("Node " << m_node->GetId() << " broadcasting beacon");
  // In real code, this would send packets to neighbors
}

void OsarMac::ReceiveBeacon(uint32_t senderId, std::vector<bool> channelState)
{
  m_neighborStates[senderId] = channelState;
}

std::map<uint32_t, std::vector<bool>> OsarMac::GetNeighborChannelStates()
{
  return m_neighborStates;
}

} // namespace ns3

