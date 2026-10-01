#include "ns3/osar-routing.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-application.h"
#include "ns3/osar-helper.h"
#include "ns3/osar-channel-model.h"
#include "ns3/osar-loss-model.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("OsarPhy");

TypeId OsarPhy::GetTypeId()
{
  static TypeId tid = TypeId("ns3::OsarPhy")
    .SetParent<Object>()
    .SetGroupName("Osar")
    .AddConstructor<OsarPhy>();
  return tid;
}

OsarPhy::OsarPhy() {}

void OsarPhy::SetNode(Ptr<Node> node)
{
  m_node = node;
}

void OsarPhy::PerformSpectrumSensing()
{
  // Simulate PU activity with exponential on/off
  m_idleSubcarriers.clear();
  for (int i = 0; i < 128; ++i)
  {
    Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
    bool idle = (uv->GetValue() > 0.3); // 70% chance idle

    m_idleSubcarriers.push_back(idle);
  }
}

std::vector<bool> OsarPhy::GetIdleSubcarriers()
{
  return m_idleSubcarriers;
}

} // namespace ns3

