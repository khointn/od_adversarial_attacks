POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -s|--searchpath)
      SEARCHPATH="$2"
      shift # past argument
      shift # past value
      ;;
    -v|--victim)
      VICTIM="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

echo "RUNNING EVALUATION WITH DATA PATH     = ${SEARCHPATH}"
echo "VICTIM     = ${VICTIM}"

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ANNOPATH="${SCRIPTPATH}/../data/coco/annotations/"

cd $SEARCHPATH
ln -s $ANNOPATH ./
mkdir val2017
cd val2017
SUCCESSPATH="$SEARCHPATH/advs_success"
FAILPATH="$SEARCHPATH/advs_fail"
NOTDETPATH="$SEARCHPATH/advs_notdet"
ln -s $SUCCESSPATH/* ./
ln -s $FAILPATH/* ./
ln -s $NOTDETPATH/* ./
echo "NUMBER OF IMAGES     = $(ls -1 | wc -l)"

cd $SCRIPTPATH/..

python evaluation.py $VICTIM --data-root "$SEARCHPATH/"
cd $SEARCHPATH && rm -rf val2017/ && rm -rf annotations