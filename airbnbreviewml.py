import emoji
import numpy as np
import pandas as pd
from numpy import mean, absolute, std
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def remove_unwanted_columns(df):
    df.drop(
        [
            "listing_url",
            "scrape_id",
            "last_scraped",
            "picture_url",
            "host_url",
            "host_thumbnail_url",
            "host_picture_url",
            "calendar_last_scraped",
            "first_review",
            "last_review",
            "license",
            "host_since",
            "host_name",
            "neighbourhood_group_cleansed",
            "calendar_updated",
            "bathrooms",
            "host_has_profile_pic",
            "has_availability",
        ],
        axis=1,
        inplace=True,
    )
    return df


dataset_listings = pd.read_csv("listings.csv")
remove_unwanted_columns(dataset_listings)
dataset_review = pd.read_csv("reviews.csv")
dataset_listings = dataset_listings.astype({'id': 'int'})
dataset_listings = dataset_listings.astype({'id': 'str'})
dataset_listings["s_listing_id"] = 'L'
dataset_listings["s_listing_id"] = dataset_listings["s_listing_id"] + dataset_listings["id"]
dataset_listings.drop(columns=['id'], inplace=True)


dataset_listings['price'] = dataset_listings['price'].str.replace('$', '')
dataset_listings['price'] = dataset_listings['price'].str.replace(',', '')
dataset_listings['host_response_rate'] = dataset_listings['host_response_rate'].str.replace('%', '')
dataset_listings[['host_response_rate']] = dataset_listings[['host_response_rate']].astype(float)
dataset_listings['host_acceptance_rate'] = dataset_listings['host_acceptance_rate'].str.replace('%', '')
dataset_listings[['host_acceptance_rate']] = dataset_listings[['host_acceptance_rate']].astype(float)
dataset_listings[['price']] = dataset_listings[['price']].astype(float)


print(dataset_listings)


dataset_review.drop(columns=['id'], inplace=True)
dataset_review = dataset_review.astype({'listing_id': 'int'})
dataset_review = dataset_review.astype({'listing_id': 'str'})
dataset_review = dataset_review.astype({'comments': 'str'})
dataset_review["s_listing_id"] = 'L'
dataset_review["s_listing_id"] = dataset_review["s_listing_id"] + dataset_review["listing_id"]
dataset_review.drop(columns=['listing_id'], inplace=True)
#print(df_review)


dataset_review = dataset_review.groupby(['s_listing_id'], as_index=False).agg({'comments': ' '.join})
df_review_count = dataset_review.groupby(['s_listing_id'], as_index = False)['comments'].count()
df_review_count = df_review_count.join(dataset_listings.set_index(["s_listing_id"]), on=["s_listing_id"], how="inner")

dataset_combined = dataset_review.join(dataset_listings.set_index(["s_listing_id"]), on=["s_listing_id"], how="inner")
dataset_combined['comments'] = dataset_combined['comments'].apply(lambda x: emoji.demojize(x, delimiters=("", "")).replace("_", " "))
dataset_combined = dataset_combined.dropna()
dataset_combined.drop(columns=['s_listing_id'], inplace=True)
print(dataset_combined)
print(dataset_combined.dtypes)


def print_feature_correlation():
    dataset_combined.neighbourhood = pd.Categorical(dataset_combined.neighbourhood)
    dataset_combined['neighbourhood'] = dataset_combined.neighbourhood.cat.codes

    dataset_combined.host_is_superhost = pd.Categorical(dataset_combined.host_is_superhost)
    dataset_combined['host_is_superhost'] = dataset_combined.host_is_superhost.cat.codes

    dataset_combined.host_location = pd.Categorical(dataset_combined.host_location)
    dataset_combined['host_location'] = dataset_combined.host_location.cat.codes

    dataset_combined.host_response_time = pd.Categorical(dataset_combined.host_response_time)
    dataset_combined['host_response_time'] = dataset_combined.host_response_time.cat.codes

    dataset_combined.host_neighbourhood = pd.Categorical(dataset_combined.host_neighbourhood)
    dataset_combined['host_neighbourhood'] = dataset_combined.host_neighbourhood.cat.codes

    dataset_combined.host_identity_verified = pd.Categorical(dataset_combined.host_identity_verified)
    dataset_combined['host_identity_verified'] = dataset_combined.host_identity_verified.cat.codes

    dataset_combined.neighbourhood_cleansed = pd.Categorical(dataset_combined.neighbourhood_cleansed)
    dataset_combined['neighbourhood_cleansed'] = dataset_combined.neighbourhood_cleansed.cat.codes

    dataset_combined.property_type = pd.Categorical(dataset_combined.property_type)
    dataset_combined['property_type'] = dataset_combined.property_type.cat.codes

    dataset_combined.room_type = pd.Categorical(dataset_combined.room_type)
    dataset_combined['room_type'] = dataset_combined.room_type.cat.codes

    dataset_combined.bathrooms_text = pd.Categorical(dataset_combined.bathrooms_text)
    dataset_combined['bathrooms_text'] = dataset_combined.bathrooms_text.cat.codes

    dataset_combined.amenities = pd.Categorical(dataset_combined.amenities)
    dataset_combined['amenities'] = dataset_combined.amenities.cat.codes

    dataset_combined.instant_bookable = pd.Categorical(dataset_combined.instant_bookable)
    dataset_combined['instant_bookable'] = dataset_combined.instant_bookable.cat.codes

    dataset_combined.isnull().sum()

    cr = dataset_combined.corr()

    max(cr[cr == 1])

    cr.host_response_time[cr.host_response_time < 1].max()

    cr2 = pd.concat([cr.iloc[1:37, 37:44], cr.iloc[44:, 37:44]])

    plt.figure(figsize=(200, 10), dpi=150)
    sns.heatmap(cr2, annot=True, cmap="YlGnBu", linewidths=.3)
    plt.xticks(rotation=30)
    plt.title("Features Co-relation Heatmap")
    plt.show()

df2 = dataset_combined.drop(
    [
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
    ],
    axis=1
)

print(df2)

Y = np.column_stack((dataset_combined["review_scores_rating"],
                     dataset_combined["review_scores_accuracy"],
                     dataset_combined["review_scores_cleanliness"],
                     dataset_combined["review_scores_checkin"],
                     dataset_combined["review_scores_communication"],
                     dataset_combined["review_scores_location"],
                     dataset_combined["review_scores_value"]))

X = df2.iloc[:, range(50)]
print(dataset_listings.dtypes.to_string())
y = dataset_combined["review_scores_rating"]

catTransformer = Pipeline(steps=[('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                 ('cat_ohe', OneHotEncoder(handle_unknown='ignore'))])
textTransformer_0 = TfidfVectorizer()
textTransformer_1 = TfidfVectorizer()
textTransformer_2 = TfidfVectorizer()
textTransformer_3 = TfidfVectorizer()
textTransformer_4 = TfidfVectorizer()

numeric_features = ['id', 'host_id', 'host_listings_count', 'host_total_listings_count',
                    'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds', 'price',
                    'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
                    'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm',
                    'maximum_nights_avg_ntm', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                    'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'host_acceptance_rate', 'host_response_rate']

categorical_features = ['source', 'host_location', 'host_response_time', 'host_is_superhost', 'host_neighbourhood', 'host_verifications',
                        'host_identity_verified', 'neighbourhood', 'neighbourhood_cleansed', 'property_type',
                        'room_type', 'bathrooms_text', 'amenities', 'instant_bookable']

review_features = ['name', 'description', 'neighborhood_overview', 'host_about', 'comments']


numTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

ct = ColumnTransformer(
    transformers=[
        ('cat', catTransformer, categorical_features),
        ('text1', textTransformer_0, 'name'),
        ('text2', textTransformer_1, 'description'),
        ('text3', textTransformer_2, 'neighborhood_overview'),
        ('text4', textTransformer_3, 'host_about'),
        ('text5', textTransformer_4, 'comments')], remainder='passthrough'
)


def performance_evaluation_lasso(X, y, ci_range):
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = []
    std_error = []
    dummy_mean_error = []

    for ci in ci_range:
        a = 1 / (2 * ci)
        model = Pipeline(steps=[('feature_engineer', ct),('LR', Lasso(alpha=a))])
        regr = MultiOutputRegressor(model, n_jobs=-1)
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(regr, X, y, scoring='neg_mean_squared_error',
                                 cv=cv, n_jobs=-1)
        dummy_model = Pipeline(steps=[('feature_engineer', ct), ('DR', DummyRegressor(strategy="mean"))])
        regr_dummy = MultiOutputRegressor(dummy_model, n_jobs=-1)
        dummy_scores = cross_val_score(regr_dummy, X, y, scoring='neg_mean_squared_error',
                                 cv=cv, n_jobs=-1)
        plotted = False
        mean_error.append(mean(absolute(scores)))
        std_error.append(std(scores))
        dummy_mean_error.append(mean(absolute(dummy_scores)))

    print("Mean of Prediction Error : ", mean_error)
    print("Standard Deviation of Prediction Error : ", std_error)
    print("Mean of Dummy Regressor : ", dummy_mean_error)
    plt.errorbar(
        ci_range, mean_error, yerr=std_error, linewidth=3, label="Lasso"
    )
    plt.xlabel("L1 Penalty (C)")
    plt.ylabel("Mean Square Error")
    plt.title("Graph of Mean Error and L1 Penalty (C)")
    plt.legend(bbox_to_anchor=(0, 1), loc='right', ncol=1)
    plt.tight_layout()
    plt.show()


def run_model_with_metrics_lasso(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=21
    )
    a = 1 / (2 * 100)
    pipeline = Pipeline(steps=[('feature_engineer', ct), ('LR', Lasso(alpha=a))])
    regr = MultiOutputRegressor(pipeline, n_jobs=-1).fit(X_train, Y_train)
    preds_test = regr.predict(X_test)
    preds_train = regr.predict(X_train)
    dummy_model = Pipeline(steps=[('feature_engineer', ct), ('DR', DummyRegressor(strategy="mean"))])
    regr_dummy = MultiOutputRegressor(dummy_model, n_jobs=-1).fit(X_train, Y_train)
    preds_dummy = regr_dummy.predict(X_test)

    print(preds_test)

    print("Test Mean Square Error ")
    print(metrics.mean_squared_error(Y_test, preds_test))
    print("Test Mean Absolute Error ")
    print(metrics.mean_absolute_error(Y_test, preds_test))
    print("Test R2 Score ")
    print(metrics.r2_score(Y_test, preds_test))
    print("Train Mean Square Error ")
    print(metrics.mean_squared_error(Y_train, preds_train))
    print("Train Mean Absolute Error ")
    print(metrics.mean_absolute_error(Y_train, preds_train))
    print("Train R2 Score ")
    print(metrics.r2_score(Y_train, preds_train))
    print("Dummy Mean Square Error ")
    print(metrics.mean_squared_error(Y_test, preds_dummy))
    print("Dummy Mean Absolute Error ")
    print(metrics.mean_absolute_error(Y_test, preds_dummy))
    print("Dummy R2 Score ")
    print(metrics.r2_score(Y_test, preds_dummy))


def run_model_with_metrics_knn(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=21
    )
    pipeline = Pipeline(steps=[('feature_engineer', ct), ('KR', KNeighborsRegressor(n_neighbors=7))])
    regr = MultiOutputRegressor(pipeline, n_jobs=-1).fit(X_train, Y_train)
    preds_test = regr.predict(X_test)
    preds_train = regr.predict(X_train)
    dummy_model = Pipeline(steps=[('feature_engineer', ct), ('DR', DummyRegressor(strategy="mean"))])
    regr_dummy = MultiOutputRegressor(dummy_model, n_jobs=-1).fit(X_train, Y_train)
    preds_dummy = regr_dummy.predict(X_test)

    print(preds_test)

    print("Test Mean Square Error ")
    print(metrics.mean_squared_error(Y_test, preds_test))
    print("Test Mean Absolute Error ")
    print(metrics.mean_absolute_error(Y_test, preds_test))
    print("Test R2 Score ")
    print(metrics.r2_score(Y_test, preds_test))
    print("Train Mean Square Error ")
    print(metrics.mean_squared_error(Y_train, preds_train))
    print("Train Mean Absolute Error ")
    print(metrics.mean_absolute_error(Y_train, preds_train))
    print("Train R2 Score ")
    print(metrics.r2_score(Y_train, preds_train))
    print("Dummy Mean Square Error ")
    print(metrics.mean_squared_error(Y_test, preds_dummy))
    print("Test Mean Absolute Error ")
    print(metrics.mean_absolute_error(Y_test, preds_dummy))
    print("Dummy R2 Score ")
    print(metrics.r2_score(Y_test, preds_dummy))


def performance_evaluation_knn(X, y, ki_range):
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = []
    std_error = []
    dummy_mean_error = []

    for n in ki_range:
        model = Pipeline(steps=[('feature_engineer', ct),('KNN', KNeighborsRegressor(n_neighbors=n))])
        regr = MultiOutputRegressor(model, n_jobs=-1)
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(regr, X, y, scoring='neg_mean_squared_error',
                                 cv=cv, n_jobs=-1)
        dummy_model = Pipeline(steps=[('feature_engineer', ct), ('DR', DummyRegressor(strategy="mean"))])
        regr_dummy = MultiOutputRegressor(dummy_model, n_jobs=-1)
        dummy_scores = cross_val_score(regr_dummy, X, y, scoring='neg_mean_squared_error',
                                 cv=cv, n_jobs=-1)
        plotted = False
        mean_error.append(mean(absolute(scores)))
        std_error.append(std(scores))
        dummy_mean_error.append(mean(absolute(dummy_scores)))

    print("Mean of Prediction Error : ", mean_error)
    print("Standard Deviation of Prediction Error : ", std_error)
    print("Mean of Dummy Regressor : ", dummy_mean_error)
    #print("Mean of Dummy Regressor : ", dummy_mean_error)
    plt.errorbar(
        ki_range, mean_error, yerr=std_error, linewidth=3, label="k-NN"
    )
    plt.xlabel("k")
    plt.ylabel("Mean Square Error")
    plt.title("Graph of Mean Square Error and k")
    plt.legend(bbox_to_anchor=(0, 1), loc='right', ncol=1)
    plt.tight_layout()
    plt.show()


ki_range = [2, 5, 7, 10, 15]
performance_evaluation_knn(X, Y, ki_range)
run_model_with_metrics_knn(X,Y)

ci_range = [1, 10, 100, 1000]
performance_evaluation_lasso(X, Y, ci_range)
run_model_with_metrics_lasso(X, Y)
print_feature_correlation()
