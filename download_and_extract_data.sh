# Download the database (if you already have, it skips this part)
if [ ! -f $"collectiona.zip" ]
then
  wget https://ibug.doc.ic.ac.uk/media/uploads/collectiona.zip
fi

if [ ! -f $"collectiona.z01" ]
then
  wget https://ibug.doc.ic.ac.uk/media/uploads/collectiona.z01
fi

if [ ! -f $"collectiona.z02" ]
then
  wget https://ibug.doc.ic.ac.uk/media/uploads/collectiona.z02
fi

if [ ! -f $"collectiona.z03" ]
then
  wget https://ibug.doc.ic.ac.uk/media/uploads/collectiona.z03
fi

if [ ! -f $"collectionb.zip" ]
then
  wget https://ibug.doc.ic.ac.uk/media/uploads/collectionb.zip
fi

if [ ! -f $"collectionb.z01" ]
then
  wget https://ibug.doc.ic.ac.uk/media/uploads/collectionb.z01
fi

#Extract the database at raw_database directory
if [ ! -d $"database" ]
then
  mkdir database
fi

zip -F collectiona.zip --out full_collectiona.zip
zip -F collectionb.zip --out full_collectionb.zip
unzip full_collectiona.zip -d database/
unzip full_collectionb.zip -d database/
rm full_collectiona.zip
rm full_collectionb.zip


