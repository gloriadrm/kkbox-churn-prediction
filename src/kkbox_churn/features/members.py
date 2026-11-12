import pandas as pd

def build_member_features(members, cutoff):
    """
    Prepara atributos de los miembros y calcula
    su antigüedad en la fecha de corte.
    """
    cutoff = pd.Timestamp(cutoff)
    members = members.copy()

    members = members.rename(columns={"bd": "age"})
    
    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    # Calcular antigüedad del cliente en días desde la fecha de registro hasta la fecha de corte 
    members["membership_age_days"] = (
        cutoff - members["registration_init_time"]
    ).dt.days

    # Seleccionar clientes con fecha de registro anterior a la fecha de corte (prevención data leakage)
    members = members.loc[
        members["membership_age_days"].isna()
        | (members["membership_age_days"] >= 0)
    ].copy()

    return members



