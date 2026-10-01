#ifndef NS3_OSAR_APPLICATION_H
#define NS3_OSAR_APPLICATION_H
#include "ns3/osar-routing.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-application.h"
#include "ns3/osar-helper.h"
#include "ns3/osar-channel-model.h"
#include "ns3/osar-loss-model.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE("OsarApplication");

TypeId OsarApplication::GetTypeId()
{
  static TypeId tid = TypeId("ns3::OsarApplication")
    .SetParent<Application>()
    .SetGroupName("Osar")
    .AddConstructor<OsarApplication>();
  return tid;
}

OsarApplication::OsarApplication() : m_packetsSent(0), m_packetsReceived(0) {}

void OsarApplication::SetNode(Ptr<Node> node)
{
  m_node = node;
}

void OsarApplication::StartApplication()
{
  NS_LOG_INFO("Application started on node " << m_node->GetId());
  m_packetsSent++;
}

void OsarApplication::StopApplication()
{
  NS_LOG_INFO("Application stopped on node " << m_node->GetId());
}

} // namespace ns3
#endif // NS3_OSAR_APPLICATION_H
