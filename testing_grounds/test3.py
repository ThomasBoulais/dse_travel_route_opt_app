import pprint

cat_dict = {
    "leisure & entertainment": [],
    "cultural, historical & religious sites": [],
    "parks, garden & nature": [],
    "sportive": [],
    "restauration": [],
    "accomodation": [],
    "transport & mobility": [],
    "utilitaries": [],
    "other": []
    }

def add_cat(cat_dict: dict, cat: str, val: str) -> dict:
    cat_dict.update({cat: cat_dict[cat] + [val]})
    return cat_dict

# "leisure & entertainment"
if True:
    add_cat(cat_dict, 'leisure & entertainment', 'LeisureComplex')
    add_cat(cat_dict, 'leisure & entertainment', 'ResidentialLeisurePark')
    add_cat(cat_dict, 'leisure & entertainment', 'BilliardRoom')
    add_cat(cat_dict, 'leisure & entertainment', 'Cabaret')
    add_cat(cat_dict, 'leisure & entertainment', 'Theater')
    add_cat(cat_dict, 'leisure & entertainment', 'Cinema')
    add_cat(cat_dict, 'leisure & entertainment', 'schema:MovieTheater')
    add_cat(cat_dict, 'leisure & entertainment', 'schema:TheaterEvent')
    add_cat(cat_dict, 'leisure & entertainment', 'HolidayCentre')
    add_cat(cat_dict, 'leisure & entertainment', 'PlayArea')
    add_cat(cat_dict, 'leisure & entertainment', 'EntertainmentAndEvent')
    add_cat(cat_dict, 'leisure & entertainment', 'schema:Event')
    add_cat(cat_dict, 'leisure & entertainment', 'ClubOrHolidayVillage')
    add_cat(cat_dict, 'leisure & entertainment', 'KidsClub')
    add_cat(cat_dict, 'leisure & entertainment', 'SportsAndLeisurePlace')
    add_cat(cat_dict, 'leisure & entertainment', 'Arena')


# "cultural, historical & religious sites"
if True:
    add_cat(cat_dict, 'cultural, historical & religious sites', 'Museum')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'TheaterEvent')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'CulturalEvent')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'CulturalActivityProvider')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'Auditorium')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'WalkingTour')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'RemarkableBuilding')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'ReligiousSite')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'ArcheologicalSite')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'RemembranceSite')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'schema:Museum')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'CulturalSite')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'RoadTour')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'TechnicalHeritage')
    add_cat(cat_dict, 'cultural, historical & religious sites', 'CastleAndPrestigeMansion')


# "parks, garden & nature"
if True:
    add_cat(cat_dict, 'parks, garden & nature', 'ParkAndGarden')
    add_cat(cat_dict, 'parks, garden & nature', 'ZooAnimalPark')
    add_cat(cat_dict, 'parks, garden & nature', 'schema:Zoo')
    add_cat(cat_dict, 'parks, garden & nature', 'schema:Park')
    add_cat(cat_dict, 'parks, garden & nature', 'Canal')
    add_cat(cat_dict, 'parks, garden & nature', 'Gorge')
    add_cat(cat_dict, 'parks, garden & nature', 'Lake')
    add_cat(cat_dict, 'parks, garden & nature', 'NaturalHeritage')
    add_cat(cat_dict, 'parks, garden & nature', 'WaterSource')
    add_cat(cat_dict, 'parks, garden & nature', 'PointOfView')
    add_cat(cat_dict, 'parks, garden & nature', 'Pond')
    add_cat(cat_dict, 'parks, garden & nature', 'Forest')
    add_cat(cat_dict, 'parks, garden & nature', 'Valley')
    add_cat(cat_dict, 'parks, garden & nature', 'Source')
    add_cat(cat_dict, 'parks, garden & nature', 'River')
    add_cat(cat_dict, 'parks, garden & nature', 'Cirque')
    add_cat(cat_dict, 'parks, garden & nature', 'UnderwaterRoute')
    add_cat(cat_dict, 'parks, garden & nature', 'Bog')
    add_cat(cat_dict, 'parks, garden & nature', 'Waterfall')
    add_cat(cat_dict, 'parks, garden & nature', 'Beach')
    add_cat(cat_dict, 'parks, garden & nature', 'CaveSinkholeOrAven')
    add_cat(cat_dict, 'parks, garden & nature', 'schema:Landform')


# "sportive"
if True:
    add_cat(cat_dict, 'sportive', 'Gymnasium')
    add_cat(cat_dict, 'sportive', 'CyclingTour')
    add_cat(cat_dict, 'sportive', 'schema:SportsEvent')
    add_cat(cat_dict, 'sportive', 'LeisureSportActivityProvider')
    add_cat(cat_dict, 'sportive', 'SwimmingPool')
    add_cat(cat_dict, 'sportive', 'SportsEvent')
    add_cat(cat_dict, 'sportive', 'SportsCompetition')
    add_cat(cat_dict, 'sportive', 'IceSkatingRink')
    add_cat(cat_dict, 'sportive', 'FitnessCenter')
    add_cat(cat_dict, 'sportive', 'TrackRollerOrSkateBoard')
    add_cat(cat_dict, 'sportive', 'EquestrianCenter')
    add_cat(cat_dict, 'sportive', 'Stables')


# "restauration"
if True:
    add_cat(cat_dict, 'restauration', 'Bakery')
    add_cat(cat_dict, 'restauration', 'FoodEstablishment')
    add_cat(cat_dict, 'restauration', 'GourmetRestaurant')
    add_cat(cat_dict, 'restauration', 'schema:FoodEstablishment')
    add_cat(cat_dict, 'restauration', 'schema:FastFoodRestaurant')
    add_cat(cat_dict, 'restauration', 'CafeOrTeahouse')
    add_cat(cat_dict, 'restauration', 'StreetFood')
    add_cat(cat_dict, 'restauration', 'BrasserieOrTavern')
    add_cat(cat_dict, 'restauration', 'schema:Restaurant')
    add_cat(cat_dict, 'restauration', 'schema:Winery')
    add_cat(cat_dict, 'restauration', 'IceCreamShop')
    add_cat(cat_dict, 'restauration', 'BarOrPub')
    add_cat(cat_dict, 'restauration', 'BistroOrWineBar')
    add_cat(cat_dict, 'restauration', 'FastFoodRestaurant')
    add_cat(cat_dict, 'restauration', 'Restaurant')
    add_cat(cat_dict, 'restauration', 'schema:IceCreamShop')
    add_cat(cat_dict, 'restauration', 'PicnicArea')
    add_cat(cat_dict, 'restauration', 'schema:Bakery')
    add_cat(cat_dict, 'restauration', 'Cellar')
    add_cat(cat_dict, 'restauration', 'TastingProvider')
    add_cat(cat_dict, 'restauration', 'schema:CafeOrCoffeeShop')
    add_cat(cat_dict, 'restauration', 'SelfServiceCafeteria')


# "accomodation"
if True:
    add_cat(cat_dict, 'accomodation', 'schema:Hotel')
    add_cat(cat_dict, 'accomodation', 'MultiPurposeRoomOrCommunityRoom')
    add_cat(cat_dict, 'accomodation', 'AccommodationProduct')
    add_cat(cat_dict, 'accomodation', 'YouthHostelAndInternationalCenter')
    add_cat(cat_dict, 'accomodation', 'SelfCateringAccommodation')
    add_cat(cat_dict, 'accomodation', 'RentalAccommodation')
    add_cat(cat_dict, 'accomodation', 'HotelRestaurant')
    add_cat(cat_dict, 'accomodation', 'schema:Hostel')
    add_cat(cat_dict, 'accomodation', 'Tipi')
    add_cat(cat_dict, 'accomodation', 'House')
    add_cat(cat_dict, 'accomodation', 'schema:House')
    add_cat(cat_dict, 'accomodation', 'Accommodation')
    add_cat(cat_dict, 'accomodation', 'Bungalow')
    add_cat(cat_dict, 'accomodation', 'Hotel')
    add_cat(cat_dict, 'accomodation', 'Room')
    add_cat(cat_dict, 'accomodation', 'TreeHouse')
    add_cat(cat_dict, 'accomodation', 'Apartment')
    add_cat(cat_dict, 'accomodation', 'FarmhouseInn')
    add_cat(cat_dict, 'accomodation', 'schema:Apartment')
    add_cat(cat_dict, 'accomodation', 'HotelTrade')
    add_cat(cat_dict, 'accomodation', 'Chalet')
    add_cat(cat_dict, 'accomodation', 'Tent')
    add_cat(cat_dict, 'accomodation', 'Guesthouse')
    add_cat(cat_dict, 'accomodation', 'ChildrensGite')
    add_cat(cat_dict, 'accomodation', 'HolidayResort')
    add_cat(cat_dict, 'accomodation', 'schema:BedAndBreakfast')
    add_cat(cat_dict, 'accomodation', 'Camping')
    add_cat(cat_dict, 'accomodation', 'CollectiveHostel')
    add_cat(cat_dict, 'accomodation', 'CamperVanArea')
    add_cat(cat_dict, 'accomodation', 'GroupLodging')
    add_cat(cat_dict, 'accomodation', 'CampingAndCaravanning')
    add_cat(cat_dict, 'accomodation', 'CollectiveAccommodation')
    add_cat(cat_dict, 'accomodation', 'StopOverOrGroupLodge')


# "transport & mobility"
if True:
    add_cat(cat_dict, 'transport & mobility', 'Transporter')
    add_cat(cat_dict, 'transport & mobility', 'TrainStation')
    add_cat(cat_dict, 'transport & mobility', 'Transport')
    add_cat(cat_dict, 'transport & mobility', 'BusStop')
    add_cat(cat_dict, 'transport & mobility', 'schema:BusStop')
    add_cat(cat_dict, 'transport & mobility', 'BikeStationOrDepot')
    add_cat(cat_dict, 'transport & mobility', 'TrainStation')
    add_cat(cat_dict, 'transport & mobility', 'TaxiCompany')
    add_cat(cat_dict, 'transport & mobility', 'TaxiStation')
    add_cat(cat_dict, 'transport & mobility', 'Parking')
    add_cat(cat_dict, 'transport & mobility', 'schema:TrainStation')
    add_cat(cat_dict, 'transport & mobility', 'CarpoolArea')
    add_cat(cat_dict, 'transport & mobility', 'ElectricVehicleChargingPoint')


# "utilitaries"
if True:
    add_cat(cat_dict, 'utilitaries', 'MedicalPlace')
    add_cat(cat_dict, 'utilitaries', 'ATM')
    add_cat(cat_dict, 'utilitaries', 'schema:Library')
    add_cat(cat_dict, 'utilitaries', 'Library')
    add_cat(cat_dict, 'utilitaries', 'ConventionCentre')
    add_cat(cat_dict, 'utilitaries', 'BoutiqueOrLocalShop')
    add_cat(cat_dict, 'utilitaries', 'CarOrBikeWash')
    add_cat(cat_dict, 'utilitaries', 'Store')
    add_cat(cat_dict, 'utilitaries', 'ConvenientService')
    add_cat(cat_dict, 'utilitaries', 'EquipmentRentalShop')
    add_cat(cat_dict, 'utilitaries', 'WifiHotSpot')
    add_cat(cat_dict, 'utilitaries', 'HealthcareProfessional')
    add_cat(cat_dict, 'utilitaries', 'schema:LocalBusiness')
    add_cat(cat_dict, 'utilitaries', 'GarageOrAirPump')


# "other"
if True:
    add_cat(cat_dict, 'other', 'schema:CivicStructure')
    add_cat(cat_dict, 'other', 'BusinessPlace')
    add_cat(cat_dict, 'other', 'ActivityProvider')
    add_cat(cat_dict, 'other', 'DefenceSite')
    add_cat(cat_dict, 'other', 'ServiceArea')
    add_cat(cat_dict, 'other', 'Producer')
    add_cat(cat_dict, 'other', 'PlaceOfInterest')
    add_cat(cat_dict, 'other', 'PointOfInterest')
    add_cat(cat_dict, 'other', 'ServiceProvider')
    add_cat(cat_dict, 'other', 'olo:OrderedList')
    add_cat(cat_dict, 'other', 'PublicLavatories')
    add_cat(cat_dict, 'other', 'Col')
    add_cat(cat_dict, 'other', 'FreePractice')
    add_cat(cat_dict, 'other', 'Practice')
    add_cat(cat_dict, 'other', 'AccompaniedPractice')
    add_cat(cat_dict, 'other', 'Product')
    add_cat(cat_dict, 'other', 'schema:Product')
    add_cat(cat_dict, 'other', 'Causse')
    add_cat(cat_dict, 'other', 'Traineeship')
    add_cat(cat_dict, 'other', 'Tour')
    add_cat(cat_dict, 'other', 'Course')

pprint.pprint(cat_dict)