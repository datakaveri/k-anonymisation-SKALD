class CategoricalGeneralizer:
    def __init__(self):
        """
        Initializes hierarchies for generalizing categorical attributes.
        """
        self.hierarchies = {
            "Blood Group": {
                1: {
                    "A+": "A+", "A-": "A-", "B+": "B+", "B-": "B-",
                    "AB+": "AB+", "AB-": "AB-", "O+": "O+", "O-": "O-"
                },
                2: {
                    "A+": "A", "A-": "A", "B+": "B", "B-": "B",
                    "AB+": "AB", "AB-": "AB", "O+": "O", "O-": "O"
                },
                3: {
                    "A+": "*", "A-": "*", "B+": "*", "B-": "*",
                    "AB+": "*", "AB-": "*", "O+": "*", "O-": "*"
                }
            },
            "GENDER": {
                1: {"Male": "Male", "Female": "Female"},
                2: {"Male": "*", "Female": "*"}
            },
            "Profession": {
                1: lambda x: x,
                2: {
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
                3: {
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
                4: lambda x: "*"
            }
        }

    # ----------------------------------------------------
    # Core generalization logic with robust error handling
    # ----------------------------------------------------
    def generalize(self, column_name, value, level):
        """
        Generalizes a categorical value with full safety checks.

        Args:
            column_name (str)
            value (str or None)
            level (int)

        Returns:
            str: generalized value or safe fallback.
        """

        # Protect against None values
        if value is None:
            return "Unknown"

        # Ensure string
        try:
            value = str(value)
        except Exception:
            return "Invalid"

        # Missing column hierarchy → return original value
        if column_name not in self.hierarchies:
            return value  

        hierarchy = self.hierarchies[column_name]

        # Invalid level
        if level not in hierarchy:
            # fallback to closest available level OR return original
            safe_level = max([lvl for lvl in hierarchy.keys() if isinstance(lvl, int)], default=None)
            if safe_level:
                level = safe_level
            else:
                return value

        mapping_or_func = hierarchy.get(level)

        # Dictionary mapping
        if isinstance(mapping_or_func, dict):
            return mapping_or_func.get(value, "Other")

        # Functional mapping
        if callable(mapping_or_func):
            try:
                return mapping_or_func(value)
            except Exception:
                return "Other"

        # Unexpected structure
        return value

    # ----------------------------------------------------
    # Convenience wrappers
    # ----------------------------------------------------
    def generalize_blood_group(self, value, level):
        return self.generalize("Blood Group", value, level)

    def generalize_profession(self, value, level):
        return self.generalize("Profession", value, level)

    def generalize_gender(self, value, level):
        return self.generalize("GENDER", value, level)
