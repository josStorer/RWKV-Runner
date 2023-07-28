# https://github.com/magenta/magenta-js/issues/164

import json
import os
import urllib.request


def get_pitches_array(min_pitch, max_pitch):
    return list(range(min_pitch, max_pitch + 1))


base_url = 'https://storage.googleapis.com/magentadata/js/soundfonts'
soundfont_path = 'sgm_plus'
soundfont_json_url = f"{base_url}/{soundfont_path}/soundfont.json"

# Download soundfont.json
soundfont_json = ""

if not os.path.exists('soundfont.json'):
    try:
        with urllib.request.urlopen(soundfont_json_url) as response:
            soundfont_json = response.read()

        # Save soundfont.json
        with open('soundfont.json', 'wb') as file:
            file.write(soundfont_json)

    except:
        print("Failed to download soundfont.json")

else:
    # If file exists, get it from the file system
    with open('soundfont.json', 'rb') as file:
        soundfont_json = file.read()

# Parse soundfont.json
soundfont_data = json.loads(soundfont_json)

if soundfont_data is not None:

    # Iterate over each instrument
    for instrument_id, instrument_name in soundfont_data['instruments'].items():

        if not os.path.isdir(instrument_name):

            # Create instrument directory if it doesn't exist
            os.makedirs(instrument_name)

        instrument_json = ""

        instrument_path = f"{soundfont_path}/{instrument_name}"

        if not os.path.exists(f"{instrument_name}/instrument.json"):

            # Download instrument.json
            instrument_json_url = f"{base_url}/{instrument_path}/instrument.json"

            try:
                with urllib.request.urlopen(instrument_json_url) as response:
                    instrument_json = response.read()

                # Save instrument.json
                with open(f"{instrument_name}/instrument.json", 'wb') as file:
                    file.write(instrument_json)

            except:
                print(f"Failed to download {instrument_name}/instrument.json")

        else:

            # If file exists, get it from the file system
            with open(f"{instrument_name}/instrument.json", 'rb') as file:
                instrument_json = file.read()

        # Parse instrument.json
        instrument_data = json.loads(instrument_json)

        if instrument_data is not None:
            # Iterate over each pitch and velocity
            for velocity in instrument_data['velocities']:

                pitches = get_pitches_array(instrument_data['minPitch'], instrument_data['maxPitch'])

                for pitch in pitches:

                    # Create the file name
                    file_name = f'p{pitch}_v{velocity}.mp3'

                    # Check if the file already exists
                    if os.path.exists(f"{instrument_name}/{file_name}"):
                        pass
                        #print(f"Skipping {instrument_name}/{file_name} - File already exists")

                    else:

                        # Download pitch/velocity file
                        file_url = f"{base_url}/{instrument_path}/{file_name}"

                        try:
                            with urllib.request.urlopen(file_url) as response:
                                file_contents = response.read()

                            # Save pitch/velocity file
                            with open(f"{instrument_name}/{file_name}", 'wb') as file:
                                file.write(file_contents)

                            print(f"Downloaded {instrument_name}/{file_name}")

                        except:
                            print(f"Failed to download {instrument_name}/{file_name}")

        else:
            print(f"Failed to parse instrument.json for {instrument_name}")

else:
    print('Failed to parse soundfont.json')