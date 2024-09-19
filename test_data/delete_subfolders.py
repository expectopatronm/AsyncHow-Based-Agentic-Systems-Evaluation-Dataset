import os
import shutil

import os
import shutil
import ast


def delete_subfolders_from_string(parent_folder, subfolder_strings):
    """
    Deletes subfolders from a list of folder names that are provided as strings.

    Args:
        parent_folder (str): The path to the parent folder where subfolders are located.
        subfolder_strings (list of str): A list of strings, each representing a list of subfolder names to be deleted.

    Returns:
        None
    """
    for subfolder_string in subfolder_strings:
        # Convert string representation of list to an actual list
        subfolder_list = ast.literal_eval(subfolder_string)[0]



        for subfolder_name in subfolder_list:
            subfolder_path = os.path.join(parent_folder, subfolder_name)
            if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                shutil.rmtree(subfolder_path)
                print(f"Deleted subfolder: {subfolder_path}")
            else:
                print(f"Subfolder not found or not a directory: {subfolder_path}")


# Example usage
parent_folder =  "tools_seq_50"
subfolder_strings = [
    "[['look_for_tech_companies_that_are_hiring', 'choose_a_tech_company', 'send_the_resume', 'drive_to_the_company', 'be_at_the_interview', 'accept_the_job']]",
    "[['go_to_the_car', 'drive_to_the_store', 'walk_in_the_store', 'go_to_produce', 'pick_up_some_lemons', 'pay_the_cashier', 'drive_back_home']]",
    "[['decide_on_venue', 'make_list_of_guests', 'send_out_invitations', 'wait_for_replies', 'check_availability_of_venue', 'book_venue_for_shower', 'send_confirmation_details_to_guests']]",
    "[['start_with_large_peas', 'rinse_the_peas', 'steam_until_wrinkled', 'add_melted_butter_cream_salt_and_pepper', 'mash_until_soft', 'eat_immediately']]",
    "[['get_dressed_and_put_on_shoes', 'grab_house_keys', 'lock_up_the_house_and_head_out', 'walk_to_the_park', 'arrive_at_the_park']]",
    "[['add_ice_to_shaker', 'add_vodka_and_liqueur', 'stir_ingredients', 'strain_into_glass', 'garnish_with_pear_slice', 'serve_drink']]",
    "[['divide_into_teams', 'prepare_to_play', 'play_the_game', 'win_the_game']]",
    "[['establish_topic_statement', 'include_main_points', 'place_main_ideas_end']]",
    "[['save_up_money_for_vacation', 'research_vacation_spots', 'choose_a_vacation_spot', 'choose_a_time_to_take_off_work', 'request_time_off_of_work']]",
    "[['go_to_the_store', 'walk_inside_the_store', 'find_the_oreo_section', 'grab_a_bag_of_oreos', 'head_to_the_checkout', 'pay_at_the_register']]",
    "[['approach_the_car', 'take_out_the_key_to_the_car', 'unlock_the_car_door', 'grab_onto_the_car_door_handle', 'pull_the_handle_forward_to_open_the_door', 'bend_down_to_sit_in_vehicles_seat', 'close_car_door']]",
    "[['practice_with_drummer', 'hit_big_note_on_one', 'hit_one_and_three', 'fill_gaps_between_one_and_three']]",
    "[['lay_out_livers', 'look_for_connective_tissue', 'trim_meat_away']]",
    "[['place_towels_at_foot_of_fireplace', 'vacuum_fireplace', 'wipe_fireplace_with_microfiber_cloth']]",
    "[['tailor_resume_to_job', 'break_resume_into_subsections', 'prioritize_key_information']]"
]

delete_subfolders_from_string(parent_folder, subfolder_strings)
