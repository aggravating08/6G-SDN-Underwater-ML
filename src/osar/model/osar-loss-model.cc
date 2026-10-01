#include "ns3/osar-routing.h"
#include "ns3/osar-mac.h"
#include "ns3/osar-phy.h"
#include "ns3/osar-application.h"
#include "ns3/osar-helper.h"
#include "ns3/osar-channel-model.h"
#include "ns3/osar-loss-model.h"
#include "ns3/log.h"
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("OsarLossModel");

TypeId OsarLossModel::GetTypeId()
{
  static TypeId tid = TypeId("ns3::OsarLossModel")
    .SetParent<Object>()
    .SetGroupName("Osar")
    .AddConstructor<OsarLossModel>();
  return tid;
}

OsarLossModel::OsarLossModel() {}

double OsarLossModel::GetPathLoss(double d, double f)
{
  double k = 1.5; // path loss exponent
  double a = 0.11 * f * f / (1 + f * f) + 44 * f * f / (4100 + f * f) + 2.75e-4 * f * f + 0.003;
  return pow(d, k) * pow(10, a / 10);
}

double OsarLossModel::GetNoisePower(double f, double w, double s)
{
  double Nt = 17 - 30 * log10(f);
  double Ns = 40 + 20 * (s - 0.5) + 26 * log10(f) - 60 * log10(f + 0.03);
  double Nw = 50 + 7.5 * sqrt(w) + 20 * log10(f) - 40 * log10(f + 0.4);
  double Nth = -15 + 20 * log10(f);
  return Nt + Ns + Nw + Nth;
}

} // namespace ns3
