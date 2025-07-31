class CategoricalGeneralizer:
    def __init__(self):
        """
        Initializes hierarchies for generalizing categorical attributes.
        """
        self.hierarchies = {
            "Blood Group": {
                1: {  # Level 0: No generalization
                    "A+": "A+", "A-": "A-", "B+": "B+", "B-": "B-",
                    "AB+": "AB+", "AB-": "AB-", "O+": "O+", "O-": "O-"
                },
                2: {  # Level 1: Group by blood type
                    "A+": "A", "A-": "A", "B+": "B", "B-": "B",
                    "AB+": "AB", "AB-": "AB", "O+": "O", "O-": "O"
                },
                3: {  # Level 2: Fully generalized
                    "A+": "*", "A-": "*", "B+": "*", "B-": "*",
                    "AB+": "*", "AB-": "*", "O+": "*", "O-": "*"
                }
            },
            "Gender": {
                1: {  # Level 0: No generalization
                    "Male": "Male", "Female" :"Female"
                },
                2: {  # Level 1: Fully generalized
                    "Male": "*", "Female" :"*"
                }
            },
            "Profession": {
                1: lambda x: x,  # Level 0: No generalization
                2: {  # Level 1: Profession → Subcategory
                    "Medical Specialists": "Healthcare",
                    "Allied Health": "Healthcare",
                    "Nursing": "Healthcare",
                    "Healthcare Support": "Healthcare",
                    "K-12 Education Teacher": "Education",
                    "Higher Education Teacher": "Education",
                    "Supplemental Education Teacher": "Education",
                    "University Professor": "Education",
                    "Performing Arts": "Creative",
                    "Visual & Media Arts": "Creative",
                    "Design": "Creative",
                    "Mixed Media Artist": "Creative",
                    "Traditional Engineering": "Engineering",
                    "Software Engineering": "Engineering",
                    "Data & Analytics": "Engineering",
                    "AI & Machine Learning": "Engineering"
                },
                3: {  # Level 2: Subcategory → Broad Category
                    "Medical Specialists": "Service Sector",
                    "Allied Health": "Service Sector",
                    "Nursing": "Service Sector",
                    "Healthcare Support": "Service Sector",
                    "K-12 Education Teacher": "Service Sector",
                    "Higher Education Teacher": "Service Sector",
                    "Supplemental Education Teacher": "Service Sector",
                    "University Professor": "Service Sector",
                    "Performing Arts": "Non-Service",
                    "Visual & Media Arts": "Non-Service",
                    "Design": "Non-Service",
                    "Mixed Media Artist": "Non-Service",
                    "Traditional Engineering": "Non-Service",
                    "Software Engineering": "Non-Service",
                    "Data & Analytics": "Non-Service",
                    "AI & Machine Learning": "Non-Service"
                },
                4: lambda x: "*"  # Level 3: Fully generalized
            }
        }

    def generalize(self, column_name, value, level):
        """
        Generalizes a categorical value based on the specified column and hierarchy level.

        Args:
            column_name (str): The name of the column (e.g., "Blood Group" or "Profession").
            value (str): The categorical value to generalize.
            level (int): The generalization level.

        Returns:
            str: The generalized value.
        """
        if column_name not in self.hierarchies:
            return value  # Return original value if column is not defined in hierarchies

        hierarchy = self.hierarchies[column_name]

        if isinstance(hierarchy.get(level), dict):  # Use dictionary-based hierarchy
            return hierarchy[level].get(value, "Other")  # Fallback to "Other" for unknown values
        elif callable(hierarchy.get(level)):  # Use function-based hierarchy
            return hierarchy[level](value)
        else:
            return value  # Return original value if no hierarchy is defined for the level

    def generalize_blood_group(self, value, level):
        """
        Wrapper for generalizing blood group values.

        Args:
            value (str): The blood group value.
            level (int): The generalization level.

        Returns:
            str: The generalized blood group value.
        """
        return self.generalize("Blood Group", value, level)

    def generalize_profession(self, value, level):
        """
        Wrapper for generalizing profession values.

        Args:
            value (str): The profession value.
            level (int): The generalization level.

        Returns:
            str: The generalized profession value.
        """
        return self.generalize("Profession", value, level)
    
    def generalize_gender(self, value, level):
        """
        Wrapper for generalizing blood group values.

        Args:
            value (str): The blood group value.
            level (int): The generalization level.

        Returns:
            str: The generalized blood group value.
        """
        return self.generalize("Gender", value, level)
