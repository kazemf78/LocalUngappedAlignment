num_seqs() {
    echo $(( $(wc -l $1 | awk '{print $1}') / 2 ))
}

DB_DIR="$HOME/dbtmp"
cd $DB_DIR

SCRIPT_DIR="$HOME/mydbs_mmseqs"
SCRIPT_PATH="$SCRIPT_DIR/prefilter2.batch"
LOG_DIR="$HOME/logtmp"

query="$HOME/data/bulk_query.fasta"
targets="$HOME/data/afdb_proteome.fasta"
query="$HOME/data/bench_tiny_query2.fasta"
query="$HOME/data/bench_one_query.fasta"
targets="$HOME/mmseqs2-benchmark-pub/db/targetannotation.fasta"

if [ ! -z $1 ]; then
    query=$1
    if [ ! -z $2 ]; then
      targets=$2
    fi
fi

echo $cpuprof $suffix
cd "$DB_DIR"

q_num_seqs=$(num_seqs $query)
t_num_seqs=$(num_seqs $targets)
time_="$(date +"%Y-%m-%d_%H-%M-%S")"
echo $time_ > "$HOME/lastrun"


if [ -z "$cpuprof" ]; then
    cpuprof="1 8"
fi

if [ -z "$suffix" ]; then
    suffix="#${cpuprof// /_}#"
fi

if [ -z ${version+x} ]; then
    version="4"
fi


module unload mmseqs2/11-e1a1c
if [ ! -x "$(command -v mmseqs)" ]; then
    echo "MMseqs local executable is not found!"
    read -p "Want to run just with the cluster module (Y/y) or exit? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
    fi
fi

module load mmseqs2/11-e1a1c
which mmseqs

for cp in $cpuprof; do
    LOG_FILE="$LOG_DIR/zout-cores=$cp-su=$suffix-$q_num_seqs-$t_num_seqs-v=$version-cluster.log"
    SUFFIX="$suffix-cores=$cp-cluster"
    echo "LOG in $LOG_FILE and DB in $DB_DIR/<DB_PREFIX>$SUFFIX"
    cp="$cp" suffix="$SUFFIX" sbatch -c $cp -o $LOG_FILE --mem=256G $SCRIPT_PATH "$query" "$targets" "${@:3}"
    echo
done


module unload mmseqs2/11-e1a1c
which mmseqs

for cp in $cpuprof; do
    LOG_FILE="$LOG_DIR/zout-cores=$cp-su=$suffix-$q_num_seqs-$t_num_seqs-v=$version-local.log"
    SUFFIX="$suffix-cores=$cp-local"
    echo "LOG in $LOG_FILE and DB in $DB_DIR/<DB_PREFIX>$SUFFIX"
    cp="$cp" suffix="$SUFFIX" sbatch -c $cp -o $LOG_FILE --mem=256G $SCRIPT_PATH "$query" "$targets" "${@:3}"
    echo
done