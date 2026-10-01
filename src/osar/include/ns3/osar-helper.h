#ifndef OSAR_HELPER_H
#define OSAR_HELPER_H

#include "ns3/node-container.h"

namespace ns3 {

class OsarHelper
{
public:
  static void Install(NodeContainer nodes);
};

} // namespace ns3

#endif // OSAR_HELPER_H

