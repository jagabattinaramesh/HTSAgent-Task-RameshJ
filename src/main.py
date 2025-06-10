#!/usr/bin/env python3
import os
from duty_calculator import DutyCalculator
from rag_qa import RAGQA

def main():
    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "../data")
    csv_dir = os.path.join(data_dir, "csvs")
    pdf_path = os.path.join(data_dir, "General Notes.pdf")
    model_path = os.path.abspath(
        os.path.join(base_dir, "../models/llama-2-7b-chat.Q4_K_M.gguf")
    )

    # Initialize components
    dc = DutyCalculator(csv_dir=csv_dir)
    rag = RAGQA(pdf_path=pdf_path, model_path=model_path)

    # CLI loop
    print("\n=== HTS AI Agent ===")
    while True:
        print("\nSelect an option:")
        print(" 1) HTS PDF QA")
        print(" 2) Calculate duties from full query")
        print(" 3) Exit")
        choice = input("> ").strip()

        if choice == "1":
            query = input("Enter your question: ")
            answer = rag.ask(query)
            print(f"\nAnswer:\n{answer}")

        elif choice == "2":
            full_query = input(
                "Enter full duty query (include HTS code, cost, freight, insurance, weight, qty):\n> "
            )
            result = dc.calculate_from_query(full_query)
            print(f"\n{result}")

        elif choice == "3":
            print("Exiting. Goodbye!")
            break

        else:
            print("Invalid choice, please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
