#ifndef OSAR_APPLICATION_H
#define OSAR_APPLICATION_H

#include "ns3/application.h"
#include "ns3/node.h"

namespace ns3 {

class OsarApplication : public Application
{
public:
  static TypeId GetTypeId();
  OsarApplication();
  void SetNode(Ptr<Node> node);
  void StartApplication() override;
  void StopApplication() override;

private:
  Ptr<Node> m_node;
  uint32_t m_packetsSent;
  uint32_t m_packetsReceived;
};

} // namespace ns3

#endif // OSAR_APPLICATION_H

