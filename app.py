import streamlit as st
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="MSL Buddy Matcher", layout="wide")

st.title("MSL Buddy Matching")
st.write("Upload the survey responses to run the matching algorithm.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataframe from the upload
    df = pd.read_csv(uploaded_file)
    
    # Validation check for required columns
    # Note: Added 'Email Address' to the required columns check
    required_cols = ['What is your full name?', 'How often can you meet with your buddy?', 'What is your email?']
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"The uploaded CSV is missing required columns. Ensure it contains: {', '.join(required_cols)}")
    else:
        with st.spinner('Running matching algorithm...'):
            # --- Pre-processing: Map names to emails ---
            name_to_email = df.set_index('What is your full name?')['What is your email?'].to_dict()

            # --- Step 1: Group based on availability ---
            availability_cols = [col for col in df.columns if "available" in col]
            melted = df.melt(id_vars=['What is your full name?'], value_vars=availability_cols)
            melted = melted.dropna(subset=['value'])
            melted['value'] = melted['value'].str.split(', ')
            exploded = melted.explode('value')

            slots_to_people = exploded.groupby(['variable', 'value'])['What is your full name?'].apply(set).to_dict()
            person_to_slots = exploded.groupby('What is your full name?').apply(
                lambda x: list(zip(x['variable'], x['value']))
            ).to_dict()

            def find_buddies(name):
                if name not in person_to_slots: return ""
                my_slots = person_to_slots[name]
                potential_buddies = set()
                for slot in my_slots:
                    potential_buddies.update(slots_to_people.get(slot, set()))
                potential_buddies.discard(name)
                return ", ".join(sorted(list(potential_buddies)))

            df['Similar Availability'] = df['What is your full name?'].apply(find_buddies)

            # --- Step 2: Meeting Frequency ---
            name_to_freq = df.set_index('What is your full name?')['How often can you meet with your buddy?'].to_dict()

            def find_similar_meeting(row):
                my_name = row['What is your full name?']
                my_frequency = row['How often can you meet with your buddy?']
                potential_buddies = str(row['Similar Availability']).split(', ') if row['Similar Availability'] else []
                matched_buddies = [b for b in potential_buddies if name_to_freq.get(b) == my_frequency]
                return ", ".join(sorted(matched_buddies))

            df['Similar Meeting'] = df.apply(find_similar_meeting, axis=1)

            # --- Step 3 & 4: Mentors/Mentees ---
            experience_map = df.set_index('What is your full name?')['What shop(s) do you have experience in?'].str.split(', ').to_dict()
            interests_map = df.set_index('What is your full name?')['What shop(s) would you like to be more involved in?'].str.split(', ').to_dict()

            def find_mentors(row):
                target_interested = set(str(row['What shop(s) would you like to be more involved in?']).split(', '))
                similar_meeting_list = str(row['Similar Meeting']).split(', ') if row['Similar Meeting'] else []
                mentors = [name for name in similar_meeting_list if target_interested.intersection(set(experience_map.get(name, [])))]
                return ", ".join(mentors)

            def find_mentees(row):
                my_experience = set(str(row['What shop(s) do you have experience in?']).split(', '))
                buddies = str(row['Similar Meeting']).split(', ') if row['Similar Meeting'] else []
                mentees = [name for name in buddies if my_experience.intersection(set(interests_map.get(name, [])))]
                return ", ".join(mentees)

            df['Potential Mentors'] = df.apply(find_mentors, axis=1)
            df['Potential Mentees'] = df.apply(find_mentees, axis=1)

            def combine_potentials(row):
                mentors = set(row['Potential Mentors'].split(', ')) if row['Potential Mentors'] else set()
                mentees = set(row['Potential Mentees'].split(', ')) if row['Potential Mentees'] else set()
                return (mentors | mentees) - {''}

            df['all_potentials'] = df.apply(combine_potentials, axis=1)

            # --- Step 5: Matching Logic ---
            sorted_names = df.sort_values(by='all_potentials', key=lambda x: x.apply(len))['What is your full name?'].tolist()
            name_to_potentials = df.set_index('What is your full name?')['all_potentials'].to_dict()
            groups, assigned = [], set()

            for name in sorted_names:
                if name in assigned: continue
                candidates = [p for p in name_to_potentials.get(name, set()) if p not in assigned]
                if candidates:
                    buddy = min(candidates, key=lambda x: len(name_to_potentials.get(x, set())))
                    groups.append([name, buddy])
                    assigned.update([name, buddy])

            preferred_size = df.set_index('What is your full name?')['What group size do you prefer?'].to_dict()
            leftovers = [n for n in df['What is your full name?'] if n not in assigned]

            for name in leftovers:
                joined = False
                for group in groups:
                    if len(group) == 2:
                        group_prefs = [preferred_size.get(member) for member in group]
                        if "A group of three" in group_prefs or "No preference" in group_prefs:
                            group.append(name)
                            assigned.add(name)
                            joined = True
                            break
                if not joined:
                    for group in groups:
                        if len(group) == 2:
                            group.append(name)
                            assigned.add(name)
                            joined = True
                            break

            # --- Build Final Results Table ---
            final_rows = []
            for i, group in enumerate(groups):
                group_id = f"Group {i+1}"
                
                for member in group:
                    # Fetching individual data from the original dataframe for this specific member
                    member_data = df[df['What is your full name?'] == member].iloc[0]
                    
                    final_rows.append({
                        'Group ID': group_id,
                        'Name': member,
                        'Email': member_data['What is your email?'],
                        'Interested Workshops': member_data['What shop(s) would you like to be more involved in?'],
                        'Experienced Workshops': member_data['What shop(s) do you have experience in?']
                    })

            # Create a clean results dataframe
            results_df = pd.DataFrame(final_rows)

            # --- Display Results ---
            st.success("Matching Complete!")
            st.subheader("Final Pairings & Workshop Interests")

            # Sort by Group ID to ensure buddies are adjacent
            results_df_sorted = results_df.sort_values("Group ID")

            # --- Define the Styling Function ---
            def style_alternate_groups(data):
                """
                Assigns alternating background colors based on changes in Group ID.
                """
                # Define colors (light gray and white)
                color_1 = 'background-color: #f9f9f9'
                color_2 = 'background-color: #ffffff'
                
                # We need to find the specific rows where Group ID changes
                # creates a boolean mask [True, False, True, ...] where True is a change
                changes = data['Group ID'] != data['Group ID'].shift()
                
                # Cumulatively sum the changes to create a helper sequence [1, 1, 2, 2, 3...]
                group_num_sequence = changes.cumsum()

                # Assign color_1 if the sequence number is odd, color_2 if even
                # Applies this logic row by row (axis=1)
                styles = np.where(group_num_sequence % 2 != 0, color_1, color_2)
                
                # Construct a DataFrame of styles identical in shape to input
                style_df = pd.DataFrame(color_2, index=data.index, columns=data.columns)
                
                # Apply the calculated row styles across all columns
                for i, row_style in enumerate(styles):
                    style_df.iloc[i, :] = row_style
                
                return style_df

            # --- Apply Style and Display ---
            # IMPORTANT: We use st.table for custom row-level styling via Pandas Styler
            try:
                # 1. Apply formatting: hide the index and set column formatting
                styler = results_df_sorted.style.hide(axis="index")\
                                                 .format(na_rep="-")\
                                                 .apply(style_alternate_groups, axis=None)
                                                 
                # 2. Render using st.table (or st.dataframe, if your version supports Styler)
                # Note: st.table renders the full static table, st.dataframe includes scrolling
                st.table(styler)
            
            except Exception as e:
                # Fallback in case formatting fails
                st.write("Displaying standard view due to formatting error.")
                st.dataframe(results_df_sorted, use_container_width=True, hide_index=True)


            # --- Download Logic ---
            csv = results_df_sorted.to_csv(index=False).encode('utf-8')
            # ... (download button remains the same)
