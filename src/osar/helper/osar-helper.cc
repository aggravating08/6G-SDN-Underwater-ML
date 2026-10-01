#include "ns3/osar-helper.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-routing.h"
#include "ns3/osar-application.h"
#include "ns3/log.h"

namespace ns3 {

// ✅ Define log component only (no NS_OBJECT_ENSURE_REGISTERED)
NS_LOG_COMPONENT_DEFINE ("OsarHelper");

void
OsarHelper::Install(NodeContainer nodes)
{
  NS_LOG_UNCOND ("Installing OSAR stack on " << nodes.GetN() << " nodes.");

  for (uint32_t i = 0; i < nodes.GetN(); ++i)
  {
    Ptr<Node> node = nodes.Get(i);
    NS_LOG_INFO ("Configuring node " << i);

    Ptr<OsarPhy> phy = CreateObject<OsarPhy>();
    Ptr<OsarMac> mac = CreateObject<OsarMac>();
    Ptr<OsarRouting> routing = CreateObject<OsarRouting>();
    Ptr<OsarApplication> app = CreateObject<OsarApplication>();

    phy->SetNode(node);
    mac->SetNode(node);
    routing->SetNode(node);
    app->SetNode(node);

    node->AggregateObject(phy);
    node->AggregateObject(mac);
    node->AggregateObject(routing);
    node->AddApplication(app);
  }
}

} // namespace ns3

