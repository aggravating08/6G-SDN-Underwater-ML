#ifndef OSAR_CHANNEL_MODEL_H
#define OSAR_CHANNEL_MODEL_H

#include "ns3/object.h"

namespace ns3 {

class OsarChannelModel : public Object
{
public:
  static TypeId GetTypeId();
  OsarChannelModel();
  double GetPropagationSpeed(double depth, double salinity, double temperature);
};

} // namespace ns3

#endif // OSAR_CHANNEL_MODEL_H

