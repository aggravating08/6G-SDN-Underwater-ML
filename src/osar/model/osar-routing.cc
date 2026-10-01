#include "ns3/osar-routing.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-application.h"
#include "ns3/osar-helper.h"
#include "ns3/osar-channel-model.h"
#include "ns3/osar-loss-model.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("OsarRouting");

TypeId OsarRouting::GetTypeId()
{
  static TypeId tid = TypeId("ns3::OsarRouting")
    .SetParent<Object>()
    .SetGroupName("Osar")
    .AddConstructor<OsarRouting>();
  return tid;
}

OsarRouting::OsarRouting() {}

void OsarRouting::SetNode(Ptr<Node> node)
{
  m_node = node;
}

void OsarRouting::SelectRelay(std::vector<Ptr<Node>> neighbors)
{
  Vector myPos = m_node->GetObject<MobilityModel>()->GetPosition();
  Ptr<Node> bestRelay;
  double minDelay = 1e9;

  for (auto neighbor : neighbors)
  {
    Vector pos = neighbor->GetObject<MobilityModel>()->GetPosition();
    double depthDiff = myPos.z - pos.z;
    if (depthDiff > 0)
    {
      double delay = depthDiff / 1500.0; // acoustic speed
      if (delay < minDelay)
      {
        minDelay = delay;
        bestRelay = neighbor;
      }
    }
  }

  if (bestRelay)
  {
    NS_LOG_INFO("Selected relay: " << bestRelay->GetId());
  }
}

} // namespace ns3

