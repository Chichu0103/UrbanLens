

# 4. Define all prompt lists as shown above
import datetime
def get_basic_prompts(location, current_year):
    return  [
    f"What is the vegetation coverage of  air quality like in {location['state']}?",
    f"What is the population of {location['city']} in {current_year} or the latest previous year?",
    f"What is the literacy rate in {location['state']} in {current_year} or the latest previous year?",
    f"What is the crime summary in {location['city']} ?",
    f"What is the sex ratio in {location['state']} in {current_year} or the latest previous year?",
    f"What is the GDP of {location['state']} in {current_year} or the latest previous year?",
    f"What is the summary of  water quality like in {location['city']}?",
    f"What is the summary of  air quality like in {location['city']}?",
    f"What is the temperature range in {location['city']} ?",
    f"What is the annual rainfall in {location['city']} ?",
    f"What is the average temperature in {location['city']}? ",
    f"What is the status of internet and mobile connectivity in {location['city']} in {current_year} or the latest previous year?",
    f"What are the major ethnic groups in {location['state']}?",
    f"What are the major religions practiced in {location['state']}?",
    f"What is the unemployment rate in {location['state']} in {current_year} or the latest previous year?",
    f"What is the name and symbol of the local currency used in {location['country']}?",
    f"What is the primary language spoken in {location['state']}?"
]

def get_consumption_prompts(location, current_year):
    return [
    f"What is the total  power consumption stats in numbers of {location['state']} with proper unit ?",
    f"What is the total water consumption stats in numbers of {location['state']} with proper unit?",
    f"What is the total fuel consumption stats in numbers of {location['state']} with proper unit?"
]

def get_crime_prompts(location, current_year):
    return[
    f" Give the statistics of criminal cases of {location['city']} ?",
    
]

def get_living_conditions_prompts(location, current_year):
    return [
    f"What is the current status of housing cost  and quality and affordability in {location['city']} ?",
    f"What is the state of waste management in {location['city']} ?",
    f"What is the no of healthcare services in {location['city']} ?",
    f"What is the majoe disease affecting  {location['state']} ?",
    f"What is the number of schools , college and institutes available in {location['city']} ?",
    f"What is the latest poverty rate in {location['state']} ",
    f"What is the cost of living in {location['city']}in {current_year} or the latest previous year?",
    f"What is the current Hunger Index  for {location['country']} "
]

def get_farming_data_prompts(location, current_year):
    return [
    f"What is the total arable land area in {location['state']} currently?",
    f"What is the water usage for irrigation in {location['state']} ?",
    f"What is the current soil type in {location['state']}?",
    f"What is the percentage of the population employed in agriculture in {location['state']} in {current_year} or the latest previous year?",
    f"What types of farming technology are used in {location['state']}, and how prevalent are they?",
    f"What is the status of food supply and security in {location['state']} ?"
]

def get_economic_data_prompts(location, current_year):
    return [
    f"What is the Gross Domestic Product (GDP) of {location['state']} in {current_year} or the latest previous year?",
    f"What is the unemployment rate in {location['city']}in {current_year} or the latest previous year?",
    f"What is the median household income in {location['city']} in {current_year} or the latest previous year?",
    f"What is the population growth rate in {location['state']} in {current_year} or the latest previous year?",

    f"What is the business growth rate in {location['city']} in {current_year} or the latest previous year?",
    f"What is the inflation rate in {location['country']} in {current_year} or the latest previous year?",
    f"What is the amount of Foreign Direct Investment (FDI) in {location['state']} in {current_year} or the latest previous year?",
    f"What is the labor force participation rate in {location['state']} in {current_year} or the latest previous year?"
]

def get_historic_disasters_prompts(location, current_year):
    return  [
    f"What natural disasters have occurred in {location['state']}, and what was their impact on the area?",
    f"What were the consequences of these natural disasters on infrastructure, residents, and the environment in {location['state']}?"
]

def get_demographic_data_prompts(location, current_year):
    return[
    f"What is the age distribution of the population in {location['state']} in {current_year} or the latest previous year?",
    f"What is the gender distribution in {location['state']} in {current_year} or the latest previous year?",
    f"What is the average household size in {location['state']} in {current_year} or the latest previous year?",
    f"What is the migration rate in {location['country']} in {current_year} or the latest previous year?",
    f"What is the birth rate in {location['state']} in {current_year} or the latest previous year?",
    f"What is the mortality rate in {location['state']} in {current_year} or the latest previous year?",
    f"What is the literacy rate in {location['state']} in {current_year} or the latest previous year?"
]

def get_industrial_stats_prompts(location, current_year):
    return  [
    f"How many people are employed in industrial sectors in {location['state']} currently?",
    f"What is the total industrial output value (in monetary terms) in {location['state']} currently?",
    f"What is the industrial growth rate in {location['state']} over the past year?",
    f"What are the major industries in {location['state']}, and what percentage of the industrial sector do they represent?",
    f"What is the total energy consumption by industries in {location['state']} ?",
    f"What is the assessment of the environmental impact of industries in {location['state']}?",
    f"What is the total amount of investment in industrial projects in {location['state']} ?"
]

def get_transportation_stats_prompts(location, current_year):
    return [
    f"What is the percentage of the population using public transit in {location['city']}?",
    f"What is the average daily traffic volume on major roads in {location['city']}?",
    f"What is the average commute time for residents in {location['city']}?",
    f"What are the primary modes of transportation used in {location['city']} (e.g., car, bus, bicycle)?",
    f"What is the availability of parking spaces in {location['city']}?",
    f"What is the status of public transportation infrastructure in {location['city']}?",
    f"how much are the traffic accident  in {location['city']} ?"
]

def get_biodiversity_stats_prompts(location, current_year):
    return [
    f"What is the total number of different Plants  species recorded in {location['state']}?",
    f"What is the total number of different animals species recorded in {location['state']}?",
    f"How many endangered species are found in {location['city']}?",
    f"What are the main habitat types present in {location['city']} and their areas?",
    f"What invasive species are present in {location['city']}, and what is their impact?",
    f"What percentage of land in {location['state']} is designated as protected areas for biodiversity?",
    f"What is the current biodiversity index for {location['state']}?",
    f"What are the distribution patterns of species in {location['state']}?"
]

def get_geographic_stats_prompts(location,current_year):
    return [
    f"What are the major topographical features of {location['city']} (e.g., mountains, valleys, plains)?",
    f"What climate zones are present in {location['city']}?",
    f"What are the predominant soil types found in {location['state']}?",
    f"What major rivers, lakes, or other water bodies are located in {location['city']}?",
    f"What natural resources are available in {location['state']}?",
    f"What is the distribution of different landforms (e.g., urban areas, agricultural land, forests) in {location['state']}?"
]

# Telecom Stats Metrics
def get_telecom_stats_prompts(location,current_year):
    return [
    f"What percentage of households in {location['city']} have access to broadband internet?",
    f"What is the coverage percentage of mobile networks (3G, 4G, 5G) in {location['city']}?",
    f"What is the availability of public Wi-Fi hotspots in {location['city']}?",
    f"How many telecommunication providers operate in {location['country']}?",
    f"What is the average monthly data usage per household in {location['country']}?",
    f"What is the digital literacy rate among residents in {location['country']}?",
    f"What percentage of the population in {location['state']} owns smart devices (smartphones, tablets, etc.)?"
]

# Public Services Stats Metrics
def get_public_services_stats_prompts(location,current_year):
    return [
    f"What is the number and distribution of healthcare facilities in {location['state']}?",
    f"What is the number of schools, colleges, and universities in {location['city']}?",
    f"What is the availability of emergency services (police, fire, ambulance) in {location['state']}?",
    f"What is the availability of parks and recreational facilities in {location['city']}?",
    f"How many public libraries are there in {location['city']}, and what services do they offer?"
]

# Noise Levels Stats Metrics
def get_noise_levels_stats_prompts(location,current_year):
    return [
    f"What are the average noise levels (in decibels) in {location['city']} for residential areas?",
    f"What are the primary sources of noise pollution in {location['city']}?",
    f"How many noise complaints have been reported in{location['city']} over the past year?",
    f"What areas in {location['city']} are designated as quiet zones, and what are their noise level limits?"
]

def generate_prompts(location, current_year=datetime.datetime.now().year):
    """
    Aggregates all prompt categories into a single dictionary.

    Parameters:
        location (dict): Location details.
        current_year (int): The current year.

    Returns:
        dict: A dictionary with categories as keys and lists of prompts as values.
    """
    return {
        'Basic Information Metrics': get_basic_prompts(location, current_year),
        'Consumption Metrics': get_consumption_prompts(location, current_year),
        'Crime Metrics': get_crime_prompts(location, current_year),
        'Living Conditions Metrics': get_living_conditions_prompts(location, current_year),
        'Farming Data Metrics': get_farming_data_prompts(location, current_year),
        'Economic Data Metrics': get_economic_data_prompts(location, current_year),
        'Historic Disasters Metrics': get_historic_disasters_prompts(location, current_year),
        'Demographic Data Metrics': get_demographic_data_prompts(location, current_year),
        'Industrial Stats Metrics': get_industrial_stats_prompts(location, current_year),
        'Transportation Stats Metrics': get_transportation_stats_prompts(location, current_year),
        'Biodiversity Stats Metrics': get_biodiversity_stats_prompts(location, current_year),
        'Geographic Stats Metrics': get_geographic_stats_prompts(location, current_year),
        'Telecom Stats Metrics': get_telecom_stats_prompts(location, current_year),
        'Public Services Stats Metrics': get_public_services_stats_prompts(location, current_year),
        'Noise Levels Stats Metrics': get_noise_levels_stats_prompts(location, current_year)
        # Add additional categories as needed
    }



