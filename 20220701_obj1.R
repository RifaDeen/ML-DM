library(readxl) #to read excel

## for k determination
#install.packages("factoextra")
library(factoextra)    

#for kmeans
#install.packages("NbClust")
library(NbClust)   

# for silhouette plot
library(cluster)

#install.packages("flexclust")
library(flexclust)


# import the dataset
x <- read_excel("Whitewine_v6.xlsx")
x

# remove the last column
wine = x[, -12]
wine
plot(wine)

#check for missing data
sum(is.na(wine)) #check for missing data


###### outlier detection/removal #############

# Create separate boxplots for each feature
par(mfrow=c(3, 4))  # Set the layout to have 3 rows and 4 columns of plots
for (i in 1:ncol(wine)) {
  boxplot(wine[, i], main=colnames(wine)[i])
}
summary(wine)


# Define a function to remove outliers based on boxplot whiskers
remove_outliers <- function(data, multiplier = 1.5) {
  outliers <- sapply(data, function(x) {
    stats <- boxplot.stats(x)
    lower_bound <- stats$stats[2] - multiplier * IQR(x)
    upper_bound <- stats$stats[4] + multiplier * IQR(x)
    return(x < lower_bound | x > upper_bound)
  })
  return(outliers)
}

# Identify outliers
outliers <- remove_outliers(wine)

# Remove outliers from the dataset
cleaned_wine <- wine
cleaned_wine[outliers] <- NA  # Mark outliers as missing values

# Remove rows with any missing values
cleaned_wine <- na.omit(cleaned_wine)
cleaned_wine

plot(cleaned_wine)


# Create separate boxplots for each feature after removing outliers
par(mfrow=c(3, 4))  
for (i in 1:ncol(cleaned_wine)) {
  # Boxplot after outlier removal
  boxplot(cleaned_wine[, i], main=colnames(cleaned_wine)[i])
}


###### Scaling the Data using z scores #############

scaled_data <- as.data.frame(scale(cleaned_wine))

# View the summary statistics of the scaled data
summary(scaled_data)

# Visualize the scaled data using boxplots
par(mfrow=c(3, 4))  # Set the layout to have 3 rows and 4 columns of plots
for (i in 1:ncol(scaled_data)) {
  boxplot(scaled_data[, i], main=colnames(scaled_data)[i])
}

plot(scaled_data)


######### k number determination #########

# NBclust
set.seed(123)
nb <- NbClust(data = scaled_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans", index = "all")
nb

# Elbow
fviz_nbclust(scaled_data, kmeans, method = 'wss')

# Gap statistics
fviz_nbclust(scaled_data, kmeans, iter.max = 100, method = 'gap_stat')

# Silhouette method
fviz_nbclust(scaled_data, kmeans, method = 'silhouette')



######## k means analysis ##########

# k=2
set.seed(123)  # Set seed for reproducibility
km <- kmeans(scaled_data, centers = 2, iter.max = 100, nstart = 20)
print(km)


# Cluster centers
print("Cluster Centers:")
print(km$centers)


####### evaluation metrics ###############

WSS <- sum(km$withinss)
WSS

BSS <- sum(km$betweenss)
BSS

TSS <- sum(km$totss)
TSS

# Calculate BSS/TSS ratio
BSS_TSS_ratio <- BSS / TSS
print("BSS/TSS Ratio:")
print(BSS_TSS_ratio)


############ Cluster Plots ###############

plot(scaled_data, col = km$cluster)

#plot cluster results
fviz_cluster(km, data = scaled_data)

#install.packages("fpc")
library(fpc)
plotcluster(scaled_data, km$cluster)


####### silhouette analysis ###########

sil <- silhouette(km$cluster, dist(scaled_data))
fviz_silhouette(sil)


######### PCA ###########

# Perform PCA on the cleaned data which is not scaled
pca_result <- prcomp(scaled_data, scale = FALSE)
summary(pca_result)


# Display the eigen vectors (principal component loadings)
# Flip the sign to positive
pca_result$rotation <- -pca_result$rotation
print(pca_result$rotation)

# Display the eigen values (variances of the principal components)
# Flip the sign to positive
pca_result$sdev <- -pca_result$sdev
VE <- pca_result$sdev^2
print(VE)


# Display the cumulative proportion of variance explained
cumulative_proportion <- cumsum(VE) / sum(VE)
print(cumulative_proportion)


# Choose the number of PCs that explain at least 85% of the variance
num_pc <- which(cumulative_proportion >= 0.85)[1]
print(paste("Number of PCs chosen:", num_pc))
num_pc


##PVE Plot
explained_variance <- (pca_result$sdev^2) / sum(pca_result$sdev^2)
varPercent <- explained_variance*100
barplot(varPercent, xlab='PC', ylab='Percent Variance', names.arg=1:length(varPercent), 
        las=1, ylim=c(0, max(varPercent)), col='gray')
abline(h=1/ncol(cleaned_wine)*100, col='red') 

##Scree plots
library(ggplot2)

# Create a data frame for plotting
explained_variance_df <- data.frame(
  PC = 1:length(explained_variance),
  Variance = (explained_variance)
)
# Create a scree plot
PVEplot <- ggplot(explained_variance_df, aes(x = PC, y = Variance)) + geom_point() +
  geom_line() + xlab("Principal Component") + ylab("Proportion of Variance Explained") + ggtitle("Scree Plot")
PVEplot

cumulative_df <- data.frame(
  PC = 1:length(cumulative_proportion),
  cVariance = (cumulative_proportion)
)
cumPVE <- ggplot(cumulative_df, aes(x = PC, y = cVariance)) + geom_point() +
  geom_line() + xlab("Principal Component") + ylab(NULL) + ggtitle("Cumulative Scree Plot")
cumPVE

# Create a new dataset with the chosen PCs

#flip the sign of pc data
pca_result$x <- -pca_result$x

transformed_data <- as.data.frame(pca_result$x[, 1:num_pc])
head(transformed_data)

summary(transformed_data)


######### (PCA) k number determination #########

# NBclust
set.seed(123)
nb <- NbClust(data = transformed_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans", index = "all")
nb

# Elbow
fviz_nbclust(transformed_data, kmeans, method = 'wss')

# Gap statistics
fviz_nbclust(transformed_data, kmeans, iter.max = 100, method = 'gap_stat')

# Silhouette method
fviz_nbclust(transformed_data, kmeans, method = 'silhouette')


######## (PCA) k means analysis ##########

# k = 2
set.seed(123)  # Set seed for reproducibility
kmPCA <- kmeans(transformed_data, centers = 2,  nstart = 100)
print(kmPCA)


# Cluster centers
print("Cluster Centers:")
print(kmPCA$centers)


####### evaluation metrics ###############


WSS <- sum(kmPCA$withinss)
WSS

BSS <- sum(kmPCA$betweenss)
BSS

TSS <- sum(kmPCA$totss)
TSS

# Calculate BSS/TSS ratio
BSS_TSS_ratio <- BSS / TSS
print("BSS/TSS Ratio:")
print(BSS_TSS_ratio)


############ Cluster Plots ###############

plot(transformed_data, col = kmPCA$cluster)

#plot cluster results
fviz_cluster(kmPCA, data = transformed_data, ellipse.type = "convex")

#install.packages("fpc")
library(fpc)
plotcluster(transformed_data, kmPCA$cluster)


####### silhouette analysis ###########

sil <- silhouette(kmPCA$cluster, dist(transformed_data))
fviz_silhouette(sil)


###### Calinski-Harabasz #########


library(fpc)
# Compute the distance matrix
dist_matrix <- dist(transformed_data)

# Compute cluster statistics
cluster_stats <- cluster.stats(dist_matrix, kmPCA$cluster)

# Print the Calinski-Harabasz index
print(cluster_stats$ch)








