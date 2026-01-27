import numpy as np
import pandas as pd


def main():
    date_str = "20200603"

    # Test rmsc04 logic
    try:
        ts = pd.to_datetime(date_str)
        dt64 = ts.to_datetime64()
        print(f"dt64 type: {type(dt64)}")

        # Check if it behaves like int
        try:
            val_int = int(dt64)
            print(f"int(dt64): {val_int}")
        except Exception as e:
            print(f"int(dt64) failed: {e}")

        if isinstance(dt64, np.datetime64):
            print(f"As int64: {dt64.astype('int64')}")
            # Check unit
            print(f"Unit: {dt64.dtype}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
