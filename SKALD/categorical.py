import logging
logger = logging.getLogger("SKALD")


class CategoricalGeneralizer:
    """
    Safe categorical generalization with deterministic fallbacks.
    """

    def __init__(self):
        self.hierarchies = {
            "blood group": {
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
            "gender": {
                1: {"Male": "Male", "Female": "Female"},
                2: {"Male": "*", "Female": "*"}
            },
            "profession": {
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
                4: lambda _: "*"
            }
        }

    # ----------------------------------------------------
    # Core generalization
    # ----------------------------------------------------
    def generalize(self, column_name: str, value, level: int) -> str:
        if value is None:
            return "Unknown"

        try:
            value = str(value)
        except Exception:
            return "Invalid"

        col_key = column_name.strip().lower()
        hierarchy = self.hierarchies.get(col_key)

        if hierarchy is None:
            return value

        if not isinstance(level, int):
            return value

        valid_levels = sorted(hierarchy.keys())

        # Clamp level safely
        if level < valid_levels[0]:
            level = valid_levels[0]
        elif level > valid_levels[-1]:
            level = valid_levels[-1]

        rule = hierarchy[level]

        if isinstance(rule, dict):
            return rule.get(value, "Other")

        if callable(rule):
            try:
                return rule(value)
            except Exception:
                return "Other"

        return value

    # ----------------------------------------------------
    # Convenience wrappers
    # ----------------------------------------------------
    def generalize_blood_group(self, value, level):
        return self.generalize("blood group", value, level)

    def generalize_gender(self, value, level):
        return self.generalize("gender", value, level)

    def generalize_profession(self, value, level):
        return self.generalize("profession", value, level)
