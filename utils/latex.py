

def to_latex_table(df):
    cols = df.columns.tolist()
    cols.insert(0, df.index.name)
    header_str = ' & '.join(cols) + r' \\'

    lines = [header_str, r'\hline']

    for idx, r in df.iterrows():
        r = r.tolist()
        r.insert(0, str(idx))
        row_str = ' & '.join(r) + r' \\'
        lines.append(row_str)

    return '\n'.join(lines)


def save_tex(df, out_file):
    with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
        if isinstance(df, list):
            for e in df:
                f.write(to_latex_table(e))
                f.write('\n')
        else:
            f.write(to_latex_table(df))
