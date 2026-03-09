from pathlib import Path

DATA_DIR    = Path(__file__).parents[2] / "data"
BRONZE_DIR  = Path(__file__).parents[2] / "data/bronze"
SILVER_DIR  = Path(__file__).parents[2] / "data/silver"
GOLD_DIR    = Path(__file__).parents[2] / "data/gold"

DEFAULT_CRS = "EPSG:4326" # WGS84       => référentiel mondial
# DEFAULT_CRS = "EPSG:2154" # Lambert-93  => référentiel français

BBOX_LEFT    = 3.10 # dimensions de la boundary box, équivalent à la ville de Bédarieux dans l'exemple
BBOX_BOTTOM  = 43.55
BBOX_RIGHT   = 3.22
BBOX_TOP     = 43.67

BBOX_DIMENSIONS = [BBOX_LEFT, BBOX_BOTTOM, BBOX_RIGHT, BBOX_TOP]

# BRONZE
OSM_PLACE_NAME              = "Hérault, Occitanie, France"
DT_BRONZE_DIR               = BRONZE_DIR / "datatourisme"
DT_DUMP_URL                 = 'https://diffuseur.datatourisme.fr/webservice/e1ec2f4e53628162352a8067eb6ac3e7/071d1b42-f48c-4350-826c-e92199a99bdf'

OSM_BRONZE_DIR              = BRONZE_DIR / "osm"
OSM_BRONZE_GEOPARQUET       = OSM_BRONZE_DIR / "osm_pois.geoparquet"

BRONZE_DRIVE_GRAPHML        = BRONZE_DIR / "drive_network.graphml"
BRONZE_WALK_GRAPHML         = BRONZE_DIR / "walk_network.graphml"

DT_DUMP_PATH                = DT_BRONZE_DIR / "dt_dump_gz"
DT_DUMP_DIR                 = DT_BRONZE_DIR / "dump"
DT_INDEX_FILE               = DT_DUMP_DIR / "index.json"


# SILVER
OSM_SILVER_GEOPARQUET       = SILVER_DIR / "osm_pois.geoparquet"

SILVER_DRIVE_GRAPHML        = SILVER_DIR / "drive_network.graphml"
SILVER_WALK_GRAPHML         = SILVER_DIR / "walk_network.graphml"

DT_SILVER_GEOPARQUET        = SILVER_DIR / "datatourisme_pois.geoparquet"
DT_SILVER_CSV               = SILVER_DIR / "datatourisme_pois.csv" # debug/exploration


# GOLD
GOLD_POIS_GEOPARQUET        = GOLD_DIR / "gold_pois.geoparquet"
GOLD_POIS_CSV               = GOLD_DIR / "gold_pois.csv" # debug/exploration

GOLD_DRIVE_GRAPHML          = GOLD_DIR / "drive_network.graphml"
GOLD_WALK_GRAPHML           = GOLD_DIR / "walk_network.graphml"

DT_DICT_TYPES_DETAILED = {
    'LeisureComplex':           'leisure & entertainment',
    'ResidentialLeisurePark':   'leisure & entertainment',
    'BilliardRoom':             'leisure & entertainment',
    'Cabaret':                  'leisure & entertainment',
    'Theater':                  'leisure & entertainment',
    'Cinema':                   'leisure & entertainment',
    'schema:MovieTheater':      'leisure & entertainment',
    'schema:TheaterEvent':      'leisure & entertainment',
    'HolidayCentre':            'leisure & entertainment',
    'PlayArea':                 'leisure & entertainment',
    'EntertainmentAndEvent':    'leisure & entertainment',
    'schema:Event':             'leisure & entertainment',
    'ClubOrHolidayVillage':     'leisure & entertainment',
    'KidsClub':                 'leisure & entertainment',
    'SportsAndLeisurePlace':    'leisure & entertainment',
    'Arena':                    'leisure & entertainment',

    'Museum':                   'cultural, historical & religious events or sites',
    'TheaterEvent':             'cultural, historical & religious events or sites',
    'CulturalEvent':            'cultural, historical & religious events or sites',
    'CulturalActivityProvider': 'cultural, historical & religious events or sites',
    'Auditorium':               'cultural, historical & religious events or sites',
    'WalkingTour':              'cultural, historical & religious events or sites',
    'RemarkableBuilding':       'cultural, historical & religious events or sites',
    'ReligiousSite':            'cultural, historical & religious events or sites',
    'ArcheologicalSite':        'cultural, historical & religious events or sites',
    'RemembranceSite':          'cultural, historical & religious events or sites',
    'schema:Museum':            'cultural, historical & religious events or sites',
    'CulturalSite':             'cultural, historical & religious events or sites',
    'RoadTour':                 'cultural, historical & religious events or sites',
    'TechnicalHeritage':        'cultural, historical & religious events or sites',
    'CastleAndPrestigeMansion': 'cultural, historical & religious events or sites',

    'ParkAndGarden':        'parks, garden & nature',
    'ZooAnimalPark':        'parks, garden & nature',
    'schema:Zoo':           'parks, garden & nature',
    'schema:Park':          'parks, garden & nature',
    'Canal':                'parks, garden & nature',
    'Gorge':                'parks, garden & nature',
    'Lake':                 'parks, garden & nature',
    'NaturalHeritage':      'parks, garden & nature',
    'WaterSource':          'parks, garden & nature',
    'PointOfView':          'parks, garden & nature',
    'Pond':                 'parks, garden & nature',
    'Forest':               'parks, garden & nature',
    'Valley':               'parks, garden & nature',
    'Source':               'parks, garden & nature',
    'River':                'parks, garden & nature',
    'Cirque':               'parks, garden & nature',
    'UnderwaterRoute':      'parks, garden & nature',
    'Bog':                  'parks, garden & nature',
    'Waterfall':            'parks, garden & nature',
    'Beach':                'parks, garden & nature',
    'CaveSinkholeOrAven':   'parks, garden & nature',
    'schema:Landform':      'parks, garden & nature',

    'Gymnasium':                    'sportive',
    'CyclingTour':                  'sportive',
    'schema:SportsEvent':           'sportive',
    'LeisureSportActivityProvider': 'sportive',
    'SwimmingPool':                 'sportive',
    'SportsEvent':                  'sportive',
    'SportsCompetition':            'sportive',
    'IceSkatingRink':               'sportive',
    'FitnessCenter':                'sportive',
    'TrackRollerOrSkateBoard':      'sportive',
    'EquestrianCenter':             'sportive',
    'Stables':                      'sportive',

    'Bakery':                       'restauration',
    'FoodEstablishment':            'restauration',
    'GourmetRestaurant':            'restauration',
    'schema:FoodEstablishment':     'restauration',
    'schema:FastFoodRestaurant':    'restauration',
    'CafeOrTeahouse':               'restauration',
    'StreetFood':                   'restauration',
    'BrasserieOrTavern':            'restauration',
    'schema:Restaurant':            'restauration',
    'schema:Winery':                'restauration',
    'IceCreamShop':                 'restauration',
    'BarOrPub':                     'restauration',
    'BistroOrWineBar':              'restauration',
    'FastFoodRestaurant':           'restauration',
    'Restaurant':                   'restauration',
    'schema:IceCreamShop':          'restauration',
    'PicnicArea':                   'restauration',
    'schema:Bakery':                'restauration',
    'Cellar':                       'restauration',
    'TastingProvider':              'restauration',
    'schema:CafeOrCoffeeShop':      'restauration',
    'SelfServiceCafeteria':         'restauration',

    'schema:Hotel':                         'accomodation',
    'MultiPurposeRoomOrCommunityRoom':      'accomodation',
    'AccommodationProduct':                 'accomodation',
    'YouthHostelAndInternationalCenter':    'accomodation',
    'SelfCateringAccommodation':            'accomodation',
    'RentalAccommodation':                  'accomodation',
    'HotelRestaurant':                      'accomodation',
    'schema:Hostel':                        'accomodation',
    'Tipi':                                 'accomodation',
    'House':                                'accomodation',
    'schema:House':                         'accomodation',
    'Accommodation':                        'accomodation',
    'Bungalow':                             'accomodation',
    'Hotel':                                'accomodation',
    'Room':                                 'accomodation',
    'TreeHouse':                            'accomodation',
    'Apartment':                            'accomodation',
    'FarmhouseInn':                         'accomodation',
    'schema:Apartment':                     'accomodation',
    'HotelTrade':                           'accomodation',
    'Chalet':                               'accomodation',
    'Tent':                                 'accomodation',
    'Guesthouse':                           'accomodation',
    'ChildrensGite':                        'accomodation',
    'HolidayResort':                        'accomodation',
    'schema:BedAndBreakfast':               'accomodation',
    'Camping':                              'accomodation',
    'CollectiveHostel':                     'accomodation',
    'CamperVanArea':                        'accomodation',
    'GroupLodging':                         'accomodation',
    'CampingAndCaravanning':                'accomodation',
    'CollectiveAccommodation':              'accomodation',
    'StopOverOrGroupLodge':                 'accomodation',

    'Transporter':                  'transport & mobility',
    'TrainStation':                 'transport & mobility',
    'Transport':                    'transport & mobility',
    'BusStop':                      'transport & mobility',
    'schema:BusStop':               'transport & mobility',
    'BikeStationOrDepot':           'transport & mobility',
    'TrainStation':                 'transport & mobility',
    'TaxiCompany':                  'transport & mobility',
    'TaxiStation':                  'transport & mobility',
    'Parking':                      'transport & mobility',
    'schema:TrainStation':          'transport & mobility',
    'CarpoolArea':                  'transport & mobility',
    'ElectricVehicleChargingPoint': 'transport & mobility',

    'MedicalPlace':             'utilitaries',
    'ATM':                      'utilitaries',
    'schema:Library':           'utilitaries',
    'Library':                  'utilitaries',
    'ConventionCentre':         'utilitaries',
    'BoutiqueOrLocalShop':      'utilitaries',
    'CarOrBikeWash':            'utilitaries',
    'Store':                    'utilitaries',
    'ConvenientService':        'utilitaries',
    'EquipmentRentalShop':      'utilitaries',
    'WifiHotSpot':              'utilitaries',
    'HealthcareProfessional':   'utilitaries',
    'schema:LocalBusiness':     'utilitaries',
    'GarageOrAirPump':          'utilitaries',

    # 'schema:CivicStructure':    'other',
    # 'BusinessPlace':            'other',
    # 'ActivityProvider':         'other',
    # 'DefenceSite':              'other',
    # 'ServiceArea':              'other',
    # 'Producer':                 'other',
    # 'PlaceOfInterest':          'other',
    # 'PointOfInterest':          'other',
    # 'ServiceProvider':          'other',
    # 'olo:OrderedList':          'other',
    # 'PublicLavatories':         'other',
    # 'Col':                      'other',
    # 'FreePractice':             'other',
    # 'Practice':                 'other',
    # 'AccompaniedPractice':      'other',
    # 'Product':                  'other',
    # 'schema:Product':           'other',
    # 'Causse':                   'other',
    # 'Traineeship':              'other',
    # 'Tour':                     'other',
    # 'Course':                   'other',
}

OSM_DICT_TYPES_DETAILED = {
    'attraction':   'leisure & entertainment', 
    'hostel':       'accomodation', 
    'hotel':        'accomodation', 
    'museum':       'cultural, historical & religious events or sites', 
    # 'yes':          '', 
    'information':  'utilitaries', 
    'viewpoint':    'parks, garden & nature', 
 
    # 'fountain':             '', 
    'reception_desk':       'utilitaries', 
    'arts_centre':          'cultural, historical & religious events or sites', 
    'animal_boarding':      'sportive', 
    'bar':                  'restauration', 
    'planetarium':          'restauration', 
    'restaurant':           'restauration', 
    'cafe':                 'restauration', 
    'place_of_worship':     'cultural, historical & religious events or sites', 
 
    # 'fountain':             '', 
    'ruins':                'cultural, historical & religious events or sites', 
    'watermill':            'cultural, historical & religious events or sites', 
    'city_gate':            'cultural, historical & religious events or sites', 
    'church':               'cultural, historical & religious events or sites', 
    'building':             'cultural, historical & religious events or sites', 
    'wayside_cross':        'cultural, historical & religious events or sites', 
    'archaeological_site':  'cultural, historical & religious events or sites', 
    'tomb':                 'cultural, historical & religious events or sites', 
    'castle':               'cultural, historical & religious events or sites', 
    'monastery':            'cultural, historical & religious events or sites', 
    # 'yes':                  '', 
    'bridge':               'cultural, historical & religious events or sites', 
 
    'dance': '', 
    'nature_reserve':   'parks, garden & nature', 
    'carousel':         'parks, garden & nature', 
    'garden':           'parks, garden & nature', 
    'park':             'parks, garden & nature', 
    'sports_centre':    'sportive', 
    'bird_hide':        'parks, garden & nature', 
  
    'cave_entrance':    'parks, garden & nature', 
    'water':            'parks, garden & nature', 
    'spring':           'parks, garden & nature', 
    'rock':             'parks, garden & nature', 
    'wood':             'parks, garden & nature', 
    'cirque':           'parks, garden & nature', 
    'bare_rock':        'parks, garden & nature', 
    'sand':             'parks, garden & nature', 
    'gorge':            'parks, garden & nature', 
    'grassland':        'parks, garden & nature', 
    'sinkhole':         'parks, garden & nature', 
    'peak':             'parks, garden & nature', 
}