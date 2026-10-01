# =========================
# OSAR Module for ns-3.38
# =========================

build_lib(
  LIBNAME osar
  SOURCE_FILES
    model/osar-phy.cc
    model/osar-mac.cc
    model/osar-routing.cc
    model/osar-application.cc
    model/osar-channel-model.cc
    model/osar-loss-model.cc
    helper/osar-helper.cc
  HEADER_FILES
  include/ns3/osar-phy.h
  include/ns3/osar-mac.h
  include/ns3/osar-routing.h
  include/ns3/osar-application.h
  include/ns3/osar-channel-model.h
  include/ns3/osar-loss-model.h
  include/ns3/osar-helper.h
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libmobility}
    ${libinternet}
)

# =========================
# =========================
# Example build
# =========================
build_example(
  NAME osar-simulation
  SOURCE_FILES
    examples/osar-simulation.cc
  LIBRARIES_TO_LINK
    ${libosar}
    ${libcore}
    ${libnetwork}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
    ${libenergy}
    ${libpoint-to-point}
)

