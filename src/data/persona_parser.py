"""
persona_parser.py

PURPOSE: This file defines and creates persona objects from a csv or json
and handles exporting them.
"""

import argparse
import csv
import json


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for persona generation",
    )
    parser.add_argument(
        "--persona-data",
        type=str,
        default="./data/personas.csv",
        help="File containing personas",
    )
    parser.add_argument(
        "--save-data",
        type=str,
        default="./data/personas.json",
        help="File to save personas",
    )
    parser.add_argument("--save-persona", action="store_true", default=False)
    args = parser.parse_args()
    return args


class PersonaEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Persona):
            return Persona.to_json(obj)
        return json.JSONEncoder.default(self, obj)


class Persona:
    def __init__(self):
        self.name = None

    def to_json(self):
        print(self.__dict__)
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(bio):
        persona = Persona()
        persona.__dict__ = json.loads(bio)
        return persona

    @staticmethod
    def from_csv_line(line):
        persona = Persona()
        persona.name = line[0]
        persona.real_name = line[1]
        persona.city = line[2]
        persona.nicknames = line[3]
        persona.group = line[4]
        persona.discog = line[5]
        persona.year = line[6]

    def to_nn_input(
        self,
        use_rn=True,
        use_city=True,
        use_nn=True,
        use_group=True,
        use_discog=True,
        use_year=True,
    ):
        res = ["<name>", self.name]
        if use_rn:
            res.append("<real name>")
            res.append(self.real_name)
        if use_city:
            res.append("<city>")
            res.append(self.city)
        if use_nn:
            res.append("<nicknames>")
            res.append(self.nicknames)
        if use_group:
            res.append("<groups>")
            res.append(self.group)
        if use_discog:
            res.append("<albums>")
            res.append(self.discog)
        if use_nn:
            res.append("<year>")
            res.append(self.year)
        return " ".join(res)


def create_personas(persona_file_name):
    """
    Creates a dictionary of personas for use in data processing
    Args:
        persona_file_name: the file to load personas from .json or .csv
    Returns:
        personas: Dict[str, Persona] which maps artist names to personas
    """
    personas = {}
    with open(persona_file_name) as persona_file:

        if persona_file_name.endswith("csv"):
            csv_reader = csv.reader(persona_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    persona = Persona.from_csv_line(row)
                    personas[persona.name] = persona
                line_count += 1
        else:
            personas = json.load(persona_file_name)
    return personas


def save_personas(save_file_name, personas):
    json.dump(personas, save_file_name, cls=PersonaEncoder)


def main():
    args = _parse_args()
    personas = create_personas(args.persona_data)
    if args.save_persona:
        save_personas(args.save_persona, personas)
    return personas


if __name__ == "__main__":
    main()
