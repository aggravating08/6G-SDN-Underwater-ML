#ifndef OSAR_LOSS_MODEL_H
#define OSAR_LOSS_MODEL_H

#include "ns3/object.h"

namespace ns3 {

class OsarLossModel : public Object
{
public:
  static TypeId GetTypeId();
  OsarLossModel();
  double GetPathLoss(double distance, double frequency);
  double GetNoisePower(double frequency, double wind, double shipping);
};

} // namespace ns3

#endif // OSAR_LOSS_MODEL_H
