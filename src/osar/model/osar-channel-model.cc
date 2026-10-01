#ifndef NS3_OSAR_CHANNEL_MODEL_H
#define NS3_OSAR_CHANNEL_MODEL_H
#include "ns3/core-module.h"
#include "ns3/object.h"
#include "ns3/osar-routing.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-application.h"
#include "ns3/osar-helper.h"
#include "ns3/osar-channel-model.h"
#include "ns3/osar-loss-model.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE("OsarChannelModel");

TypeId OsarChannelModel::GetTypeId()
{
  static TypeId tid = TypeId("ns3::OsarChannelModel")
    .SetParent<Object>()
    .SetGroupName("Osar")
    .AddConstructor<OsarChannelModel>();
  return tid;
}

OsarChannelModel::OsarChannelModel() {}

double OsarChannelModel::GetPropagationSpeed(double z, double s, double T)
{
  return 1449.05 + 45.7 * T - 5.217 * z + 0.2373 + (1.333 - 0.126 * T + 0.00972) * (s - 35) + 16.3 * z + 0.18 * z * z;
}

} // namespace ns3
#endif // NS3_OSAR_CHANNEL_MODEL_H
