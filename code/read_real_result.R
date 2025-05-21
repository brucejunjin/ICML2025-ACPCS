dataset <- 'Wine' # available for 'Wine', 'Polite', 'Census'
target_index <- 1 # for Wine: 1 to 6; for Polite: 1 to 2; for Census: 1 to 3
tabnam <- 'length' # available for 'length' and 'bias'

if (tabnam == 'length'){
  filename <- paste0('../output/realdata/Wine/CIlength', target_index, '.rda')
} else {
  filename <- paste0('../output/realdata/Wine/Biaslength', target_index, '.rda')
}

load(filename)

if (tabnam == 'length'){
  print('Mean:')
  print(rowMeans(length_record))
  print('SD:')
  print(apply(X=length_record, MARGIN=1, FUN=sd))
} else {
  print('Mean:')
  print(rowMeans(bias_record))
  print('SD:')
  print(apply(X=bias_record, MARGIN=1, FUN=sd))
}

