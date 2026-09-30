def execute_query(con, query):
    try:
        result = con.execute(query).fetchdf()
        return result, None
    except Exception as e:
        return None, str(e)
