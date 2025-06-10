import os
import pandas as pd
import re


def parse_duty_advanced(duty_str: str, cif: float, wt: float = None, qty: int = None) -> float:
    """
    Parse a duty string and return the duty as a fraction of CIF.

    - "free"                -> 0.0
    - "5%"                  -> 0.05
    - "10¢/kg" with wt      -> (10 * wt / 100) / cif
    - "$2.50/unit" with qty -> (2.50 * qty) / cif
    """
    if not duty_str or not isinstance(duty_str, str):
        return 0.0
    s = duty_str.lower()
    if "free" in s:
        return 0.0

    # percentage
    m = re.search(r"([0-9.]+)%", s)
    if m:
        return float(m.group(1)) / 100.0

    # cents per kg
    m = re.search(r"([0-9.]+)¢/kg", s)
    if m and wt is not None:
        return (float(m.group(1)) * wt / 100.0) / cif

    # dollars per unit
    m = re.search(r"\$([0-9.]+)/unit", s)
    if m and qty is not None:
        return (float(m.group(1)) * qty) / cif

    return 0.0


class DutyCalculator:
    def __init__(self, csv_dir: str):
        """
        Load all CSVs from the given directory into a single DataFrame.
        Auto-detect HTS code and duty columns.
        """
        frames = []
        for fn in os.listdir(csv_dir):
            if fn.lower().endswith('.csv'):
                df = pd.read_csv(os.path.join(csv_dir, fn))
                frames.append(df)
        if not frames:
            raise ValueError(f"No CSV files found in directory: {csv_dir}")
        self.df = pd.concat(frames, ignore_index=True)

        # Detect code column
        if 'HTS_Code' in self.df.columns:
            self.code_col = 'HTS_Code'
        elif 'HTS Number' in self.df.columns:
            self.code_col = 'HTS Number'
        else:
            raise ValueError("CSV must contain 'HTS_Code' or 'HTS Number' column.")

        # Detect duty column
        if 'Duty' in self.df.columns:
            self.duty_col = 'Duty'
        else:
            rate_cols = [c for c in self.df.columns if 'Rate' in c]
            if rate_cols:
                self.duty_col = rate_cols[0]
            else:
                raise ValueError("CSV must contain 'Duty' or a 'Rate' column.")

    def calculate(self,
                  product_cost: float,
                  freight: float,
                  insurance: float,
                  hts_code: str,
                  unit_weight: float = None,
                  quantity: int = None) -> float:
        """
        Core calculation: returns duty amount in same currency as costs.
        """
        cif = product_cost + freight + insurance
        match = self.df[self.df[self.code_col] == hts_code]
        if match.empty:
            raise ValueError(f"HTS code {hts_code} not found.")
        duty_str = match.iloc[0][self.duty_col]
        frac = parse_duty_advanced(duty_str, cif, wt=unit_weight, qty=quantity)
        return frac * cif

    def calculate_from_query(self, query: str) -> str:
        """
        Parse a single natural-language query and compute the duty.
        Recognizes HTS code, cost, freight, insurance, weight (kg), and quantity in one go.
        """
        # Regex patterns
        cm = re.search(r"hts code\s*([0-9\.]+)", query, re.I)
        cc = re.search(r"(?:cost|product cost|fob cost)\s*(?:of\s*)?\$?([0-9,]+(?:\.[0-9]+)?)", query, re.I)
        fm = re.search(r"freight\s*(?:of\s*)?\$?([0-9,]+(?:\.[0-9]+)?)", query, re.I)
        im = re.search(r"insurance\s*(?:of\s*)?\$?([0-9,]+(?:\.[0-9]+)?)", query, re.I)
        wm = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*kg", query, re.I)
        qm = re.search(r"([0-9,]+)\s*units?", query, re.I)

        if not cm or not cc:
            return "❗️ Could not parse HTS code or cost from the query."

        code = cm.group(1)
        # remove commas
        cost = float(cc.group(1).replace(',', ''))
        freight = float(fm.group(1).replace(',', '')) if fm else 0.0
        insurance = float(im.group(1).replace(',', '')) if im else 0.0
        weight = float(wm.group(1).replace(',', '')) if wm else None
        qty = int(qm.group(1).replace(',', '')) if qm else None
        cif = cost + freight + insurance

        try:
            duty_amt = self.calculate(cost, freight, insurance, code, unit_weight=weight, quantity=qty)
        except ValueError as e:
            return f"❗️ {e}"

        return (
            f"**HTS Code:** {code}\n"
            f"**CIF:** ${cif:,.2f}\n"
            f"**Duty:** ${duty_amt:,.2f}"
        )
